
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
    """
    header_file: Union[str, Path]
    data_files: Sequence[Union[str, Path]]
    df: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    columns: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.header_file = Path(self.header_file)
        self.data_files = [Path(p) for p in self.data_files]
        self.columns = self._parse_header(self.header_file)
        self.df = self._load_data(self.data_files, self.columns)

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

    def list_variables(self, limit: Optional[int] = None) -> List[str]:
        cols = [c for c in self.df.columns if c != "Time"]
        return cols[:limit] if limit is not None else cols

    def available_timerange(self) -> Optional[tuple]:
        if self.df.empty:
            return None
        return (self.df.index.min(), self.df.index.max())

    def select(self, columns: Union[str, List[str]], start: Optional[Union[str, pd.Timestamp]] = None,
               end: Optional[Union[str, pd.Timestamp]] = None, resample: Optional[str] = None) -> pd.DataFrame:
        if isinstance(columns, str):
            columns = [columns]
        data = self.df.copy()
        if start or end:
            data = data.loc[start:end]
        columns = [c for c in columns if c in data.columns]
        if not columns:
            raise ValueError("None of the requested columns exist in the data.")
        sel = data[columns]
        if resample:
            sel = sel.resample(resample).mean()
        return sel
    
    def plot(self, column: Union[str, List[str]],
            start: Optional[Union[str, pd.Timestamp]] = None,
            end: Optional[Union[str, pd.Timestamp]] = None,
            resample: Optional[str] = None,
            engine: str = "matplotlib"):
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
        """
        # Normalize to a list of columns
        cols = [column] if isinstance(column, str) else list(column)
        df = self.select(cols, start, end, resample)

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
                y=df.columns,                       # plot all requested columns
                title=", ".join(df.columns) + " vs Time",
                labels={"x": "Time", "value": "Value", "variable": "Channel"},
            )
            fig.update_layout(hovermode="x unified")
            return fig

        # Matplotlib branch
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


