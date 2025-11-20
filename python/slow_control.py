from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union
import pandas as pd
import matplotlib.pyplot as plt
import re

@dataclass
class SlowControlLog:
    """
    Slow control log reader and plotting helper.

    - Header parsing is resilient to wrapped/tabbed headers and duplicates.
    - Data parsing uses European decimal comma and day-first timestamps.
    - You can resample with log.select("t0", resample="5min") to smooth plots.
    - Supports time range selection for highlighting/filtering data.
    """
    header_file: Union[str, Path]
    data_files: Sequence[Union[str, Path]]
    df: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    columns: List[str] = field(default_factory=list, init=False)
    _selection_mask: Optional[pd.Series] = field(default=None, init=False)

    def __post_init__(self):
        self.header_file = Path(self.header_file)
        self.data_files = [Path(p) for p in self.data_files]
        self.columns = self._parse_header(self.header_file)
        self.df = self._load_data(self.data_files, self.columns)
        self._selection_mask = None  # None means all data selected

    @staticmethod
    def _parse_header(header_path: Path) -> List[str]:
        """
        Parse a header text file whose column names are tab-separated but may be wrapped across lines.
        Returns a list of column names.
        """
        text = header_path.read_text(encoding="utf-8", errors="replace")
        unified = re.sub(r"\r?\n+", " ", text)        # unify newlines to spaces
        unified = re.sub(r"\t+", "\t", unified)       # collapse multiple tabs
        raw_cols = [c.strip() for c in unified.split("\t")]
        cols = [c for c in raw_cols if c]             # drop empty tokens
        # Deduplicate adjacent tokens possibly introduced by wraps
        deduped = []
        for c in cols:
            if not deduped or deduped[-1] != c:
                deduped.append(c)
        # Ensure timestamp column exists and is first
        if deduped and not deduped[0].lower().startswith("time"):
            deduped.insert(0, "Time")
        # Make names unique
        seen = {}
        unique_cols = []
        for c in deduped:
            base = c
            if c in seen:
                seen[c] += 1
                c = f"{base}__{seen[base]}"
            else:
                seen[c] = 0
            unique_cols.append(c)
        return unique_cols

    @staticmethod
    def _read_single_file(path: Path, columns: List[str]) -> pd.DataFrame:
        # Read tab-separated values with European decimal comma; skip malformed lines.
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=columns,
            engine="python",
            decimal=",",
            on_bad_lines="skip",
        )
        # Parse datetime in the first column (day-first).
        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Time"])
            df = df.set_index("Time").sort_index()
        else:
            first = df.columns[0]
            df[first] = pd.to_datetime(df[first], dayfirst=True, errors="coerce")
            df = df.dropna(subset=[first]).set_index(first).sort_index()
            df.index.name = "Time"
        return df

    @classmethod
    def _load_data(cls, data_files: Sequence[Path], columns: List[str]) -> pd.DataFrame:
        frames = []
        for p in data_files:
            if p.exists():
                frames.append(cls._read_single_file(p, columns))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=0)
        df = df[~df.index.duplicated(keep="first")]
        return df

    def set_time_ranges(self, time_ranges: Optional[List[tuple[str, str]]]):
        """
        Set time ranges to select/highlight. Pass None to select all data.
        
        Parameters
        ----------
        time_ranges : list of tuple(str, str) | None
            List of (start, end) time range tuples, or None to reset to all data.
        """
        if time_ranges is None:
            self._selection_mask = None
            self._segments = None
            return
        if self.df.empty:
            self._selection_mask = None
            self._segments = None
            return
        # Normalize and store segments explicitly
        segs = []
        mask = pd.Series(False, index=self.df.index)
        for start, end in time_ranges:
            s = pd.to_datetime(start)
            e = pd.to_datetime(end)
            if e < s:
                continue
            segs.append((s, e))
            mask |= (self.df.index >= s) & (self.df.index <= e)
        self._selection_mask = mask
        self._segments = segs  # list of (start,end)

    def list_variables(self, limit: Optional[int] = None) -> List[str]:
        cols = [c for c in self.df.columns if c != "Time"]
        return cols[:limit] if limit is not None else cols

    def available_timerange(self) -> Optional[tuple]:
        if self.df.empty:
            return None
        return (self.df.index.min(), self.df.index.max())

    def compute_stats(self,
                      columns: Optional[Union[str, List[str]]] = None,
                      start: Optional[Union[str, pd.Timestamp]] = None,
                      end: Optional[Union[str, pd.Timestamp]] = None,
                      resample: Optional[str] = None,
                      selected_only: bool = False,
                      by_segment: bool = False) -> pd.DataFrame:
        """
        Compute statistics (mean, std, count, stderr) for variables.

        Parameters
        ----------
        columns : str | list[str] | None
            Variables to include. None = all.
        start, end : optional time limits.
        resample : str | None
            Pandas offset alias (e.g. '5min'). If given, stats on resampled data.
        selected_only : bool
            Use only selected time ranges (if mask set).
        by_segment : bool
            If True and selected_only, split stats per contiguous selected segment.

        Returns
        -------
        DataFrame with columns:
            variable, mean, std, count, stderr
            (plus segment_id, segment_start, segment_end if by_segment)
        """
        # Determine variables
        if columns is None:
            cols = self.list_variables()
        else:
            cols = [columns] if isinstance(columns, str) else list(columns)
        if not cols:
            return pd.DataFrame()

        base = self.select(cols, start=start, end=end,
                           resample=resample, selected_only=selected_only)
        if base.empty:
            return pd.DataFrame()

        segments = getattr(self, "_segments", None)
        use_segments = by_segment and selected_only and segments

        rows = []

        def summarize(seg_df: pd.DataFrame, seg_id: int, seg_start, seg_end):
            out = {
                "segment_id": seg_id,
                "segment_start": seg_start,
                "segment_end": seg_end
            }
            for c in seg_df.columns:
                s = seg_df[c].dropna()
                n = int(s.count())
                if n == 0:
                    out[f"{c}_mean"] = float("nan")
                    out[f"{c}_std"] = float("nan")
                    out[f"{c}_stderr"] = float("nan")
                    out[f"{c}_count"] = 0
                    continue
                mu = float(s.mean())
                sd = float(s.std(ddof=1)) if n > 1 else 0.0
                se = sd / (n ** 0.5) if n > 0 else float("nan")
                out[f"{c}_mean"] = mu
                out[f"{c}_std"] = sd
                out[f"{c}_stderr"] = se
                out[f"{c}_count"] = n
            rows.append(out)

        if use_segments:
            for seg_id, (s, e) in enumerate(segments):
                if start and pd.to_datetime(e) < pd.to_datetime(start):
                    continue
                if end and pd.to_datetime(s) > pd.to_datetime(end):
                    continue
                seg_df = base.loc[s:e]
                if seg_df.empty:
                    continue
                summarize(seg_df, seg_id, s, e)
        else:
            summarize(base, 0, base.index.min(), base.index.max())

        df_stats = pd.DataFrame(rows)
        if "segment_id" in df_stats.columns:
            df_stats = df_stats.sort_values("segment_id").reset_index(drop=True)
        return df_stats

    def add_derived_column(self, name: str, expression):
        """
        Add a derived column to the internal dataframe.

        Parameters
        ----------
        name : str
            Name of the new column.
        expression : callable or pandas Series
            Either a function that takes the dataframe and returns a Series,
            or a pre-computed Series with matching index.

        Example
        -------
        log.add_derived_column("dp", lambda df: df["Pmain"] - df["P102"])
        log.add_derived_column("dp2", lambda df: df["Pmain"]**2 - df["P102"]**2)
        """
        if callable(expression):
            self.df[name] = expression(self.df)
        else:
            self.df[name] = expression
        # Update column list
        if name not in self.columns:
            self.columns.append(name)

    def add_derived_columns(self, column_dict: dict):
        """
        Add multiple derived columns at once.
        
        Parameters
        ----------
        column_dict : dict
            Dictionary mapping column names to expressions (callable or Series).
        
        Example
        -------
        log.add_derived_columns({
            "dp": lambda df: df["Pmain"] - df["P102"],
            "dp2": lambda df: df["Pmain"]**2 - df["P102"]**2
        })
        """
        for name, expr in column_dict.items():
            self.add_derived_column(name, expr)    

    def select(self, columns: Union[str, List[str]], start: Optional[Union[str, pd.Timestamp]] = None,
               end: Optional[Union[str, pd.Timestamp]] = None, resample: Optional[str] = None,
               selected_only: bool = False) -> pd.DataFrame:
        """
        Select one or more columns within an optional time window, with optional resampling.
        
        Parameters
        ----------
        columns : str | list[str]
            Column name or list of column names to select.
        start, end : str | pandas.Timestamp | None
            Optional time window.
        resample : str | None
            Pandas offset alias (e.g. '5min') to average over time bins.
        selected_only : bool
            If True and time ranges are set, only return data within selected ranges.
        
        Returns
        -------
        pandas.DataFrame
        """
        if isinstance(columns, str):
            columns = [columns]

        data = self.df.copy()

        # Apply time range selection if requested
        if selected_only and self._selection_mask is not None:
            data = data.loc[self._selection_mask]

        if start or end:
            data = data.loc[start:end]
        
        columns = [c for c in columns if c in data.columns]
        if not columns:
            raise ValueError("None of the requested columns exist in the data.")
        sel = data[columns]
        if resample:
            sel = sel.resample(resample).mean()
        return sel

    def get_variable(self, column: str,
                     start: Optional[Union[str, pd.Timestamp]] = None,
                     end: Optional[Union[str, pd.Timestamp]] = None,
                     resample: Optional[str] = None,
                     selected_only: bool = False) -> pd.Series:
        """
        Get a single variable as a pandas Series.
        
        Parameters
        ----------
        column : str
            Column name to retrieve.
        start, end : str | pandas.Timestamp | None          
            Optional time window.
        resample : str | None
            Pandas offset alias (e.g. '5min') to average over time bins.
        selected_only : bool
            If True and time ranges are set, only return data within selected ranges.
        
        Returns
        -------
        pandas.Series
        """
        return self.select(column, start, end, resample, selected_only).squeeze()
    
    def plot(self, column: Union[str, List[str]],
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        resample: Optional[str] = None,
        engine: str = "matplotlib",
        show_unselected: bool = True):
        """
        Plot one or many variables vs time.

        Parameters
        ----------
        column : str | list[str]
            Column name or list of column names to plot.
        start, end : str | pandas.Timestamp | None
            Optional time window.
        resample : str | None
            Pandas offset alias (e.g. '5min') to average over time bins.
        engine : {'matplotlib','plotly'}
            - 'matplotlib' (default): returns a Matplotlib Axes.
            - 'plotly': returns a Plotly Figure (zoom/pan/hover enabled).
        show_unselected : bool
            If True and time ranges are set, show unselected data in grey with selected data colored.
            If False, only show selected data.
        """
        # Normalize to a list of columns
        cols = [column] if isinstance(column, str) else list(column)

        # If we have a selection mask and plotly engine, handle grey background overlay
        if engine == "plotly" and self._selection_mask is not None and show_unselected:
            try:
                import plotly.graph_objects as go
            except Exception as e:
                raise RuntimeError(
                    "Plotly is required for engine='plotly'. Install with `pip install plotly`."
                ) from e

            fig = go.Figure()
            
            for var in cols:
                # Get full data (with resampling if requested)
                data_full = self.select(var, start, end, resample, selected_only=False).squeeze()
                
                # Resample the mask to match the resampled data
                if resample and self._selection_mask is not None:
                    mask_resampled = self._selection_mask.reindex(data_full.index, method='nearest', limit=1)
                    mask_resampled = mask_resampled.fillna(False)
                else:
                    mask_resampled = self._selection_mask.reindex(data_full.index, fill_value=False)
                
                # For unselected data: plot in grey
                data_unselected = data_full[~mask_resampled]
                if not data_unselected.empty:
                    fig.add_trace(go.Scatter(
                        x=data_unselected.index,
                        y=data_unselected.values,
                        name=f"{var} (background)",
                        line=dict(color="lightgrey", width=1),
                        showlegend=False,
                        hoverinfo='skip',
                        connectgaps=False
                    ))
                
                # For selected data: set unselected points to NaN to break lines
                data_with_gaps = data_full.copy()
                data_with_gaps[~mask_resampled] = None
                
                fig.add_trace(go.Scatter(
                    x=data_with_gaps.index,
                    y=data_with_gaps.values,
                    name=var,
                    mode='lines',
                    connectgaps=False
                ))
            
            fig.update_layout(
                title=", ".join(cols) + " vs Time",
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode="x unified"
            )
            return fig

        # For matplotlib with selection mask overlay
        if engine == "matplotlib" and self._selection_mask is not None and show_unselected:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots()
            
            for var in cols:
                # Get full data (with resampling if requested)
                data_full = self.select(var, start, end, resample, selected_only=False).squeeze()
                
                # Resample the mask to match the resampled data
                if resample and self._selection_mask is not None:
                    mask_resampled = self._selection_mask.reindex(data_full.index, method='nearest', limit=1)
                    mask_resampled = mask_resampled.fillna(False)
                else:
                    mask_resampled = self._selection_mask.reindex(data_full.index, fill_value=False)
                
                # Plot unselected in grey
                data_unselected = data_full[~mask_resampled]
                if not data_unselected.empty:
                    ax.plot(data_unselected.index, data_unselected.values, 
                           color='lightgrey', linewidth=1, label='_nolegend_')
                
                # Plot selected in color
                data_selected = data_full[mask_resampled]
                if not data_selected.empty:
                    ax.plot(data_selected.index, data_selected.values, label=var)
            
            ax.set_xlabel("Time")
            ax.set_ylabel(", ".join(cols))
            ax.set_title(", ".join(cols) + " vs Time")
            if len(cols) > 1:
                ax.legend(loc="best")
            fig.tight_layout()
            return ax

        # For non-overlay plots, use selected_only flag
        selected_only = self._selection_mask is not None and not show_unselected
        df = self.select(cols, start, end, resample, selected_only=selected_only)

        if engine == "plotly":
            try:
                import plotly.express as px
            except Exception as e:
                raise RuntimeError(
                    "Plotly is required for engine='plotly'. Install with `pip install plotly`."
                ) from e

            fig = px.line(
                df,
                x=df.index,
                y=df.columns,
                title=", ".join(df.columns) + " vs Time",
                labels={"x": "Time", "value": "Value", "variable": "Channel"},
            )
            fig.update_layout(hovermode="x unified")
            return fig

        # Matplotlib branch (no selection overlay)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if len(df.columns) == 1:
            df.iloc[:, 0].plot(ax=ax)
            ax.set_ylabel(df.columns[0])
        else:
            for c in df.columns:
                ax.plot(df.index, df[c], label=c)
            ax.legend(loc="best")
            ax.set_ylabel(", ".join(df.columns))

        ax.set_xlabel("Time")
        ax.set_title(", ".join(df.columns) + " vs Time")
        fig.tight_layout()
        return ax