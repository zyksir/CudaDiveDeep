import json
import argparse
import pandas as pd
from typing import Dict, Any, Optional
from hta.common.trace import parse_trace_file
from typing import List, Union

pd.set_option('future.no_silent_downcasting', True)
class TorchProfile:
    def __init__(self, trace_path: str):
        self.trace_path = trace_path
        self.meta, df, symbol_table = parse_trace_file(trace_path)
        sym_table = symbol_table.get_sym_table()
        df["name"] = df["name"].apply(lambda x: sym_table[x])
        df["cat"] = df["cat"].apply(lambda x: sym_table[x])
        df["end"] = df.apply(lambda row: row["ts"] + row["dur"], axis=1)

        name_dict = {
            "name": "kernel_name",
            "ts": "start_ts",
            "end": "end_ts",
            "stream": "stream_id"
        }
        df = df.rename(columns=name_dict)
        self.cpu_df = df[~df["stream_id"].gt(0)]
        self.gpu_df = df[df["stream_id"].gt(0)]
        self.gpu_df = self.gpu_df[["start_ts", "end_ts", "kernel_name", "stream_id"]].sort_values(by="start_ts")
        self.cpu_df = self.cpu_df[["start_ts", "end_ts", "kernel_name", "cat"]].sort_values(by="start_ts")
        self.trace_duration_us = self.gpu_df["end_ts"].max() - self.gpu_df["start_ts"].min()
    
    def _add_gpu_annotation(self, ignore_gpu_annotation: Optional[str]=None):
        """
        GPU user annotation is added by `torch.profiler.record_function`
        We assume that all the GPU kernels launched after the annotation and ended before the annotation are part of this annotation.
        """
        gpu_df = self.gpu_df
        cpu_df = self.cpu_df[self.cpu_df["cat"].str.contains("gpu_user_annotation")]
        annotation_delta = 2 
        if ignore_gpu_annotation is not None:
            cpu_df = cpu_df[~cpu_df.kernel_name.str.match(ignore_gpu_annotation)]

        gpu_df.sort_values(by="end_ts", inplace=True)
        cpu_df.sort_values(by="end_ts", inplace=True)
        cpu_df["duration"] = cpu_df["end_ts"] - cpu_df["start_ts"]
        gpu_annotation = [""] * len(gpu_df)
        for annotation_name in sorted(cpu_df["kernel_name"].unique()):
            annotation_df = cpu_df[cpu_df["kernel_name"].eq(annotation_name)]
            i, j = 0, 0
            while i < len(gpu_df) and j < len(annotation_df):
                if gpu_df.iloc[i]["end_ts"] < annotation_df.iloc[j]["end_ts"] + annotation_delta:
                    if gpu_df.iloc[i]["start_ts"] > annotation_df.iloc[j]["start_ts"] - annotation_delta:
                        gpu_annotation[i] += annotation_df.iloc[j]["kernel_name"] + "::"
                    i += 1
                else:
                    j += 1

        gpu_df["gpu_annotation"] = gpu_annotation
        gpu_df["gpu_annotation"] = gpu_df["gpu_annotation"].str.strip("::")
        self.gpu_df = gpu_df.sort_values(by="start_ts")
        return self.gpu_df

    def _add_tag(self, df: pd.DataFrame, rule_config: Dict[str, Any], priority_base: int = 1):
        if df.empty:
            return [df], {}
        assert "RULE_TYPE" in rule_config, f"invalid subrules {rule_config}"
        filter_rule = df["start_ts"].lt(0)
        column = rule_config["RULE_TYPE"]
        for rule in rule_config["RULES"]:
            prefix, pattern = rule["PREFIX"], rule["PATTERN"]
            if pattern is None:
                pattern = ".*"
            if prefix is None:
                prefix = ""
            sub_df = df[df[column].str.match(pattern) & ~filter_rule]
            filter_rule = filter_rule | df[column].str.match(pattern)
            if rule["PREFIX"] is not None and rule["PATTERN"] is not None:
                sub_df["tag"] += prefix + ": "
            else:
                sub_df["tag"] += sub_df[column] + ": "
            rule["SUB_DF"] = sub_df

        tagged_df_list, tag2priority = [], {}
        rule_priority = rule_config.get("PRIORITY", [])
        rule_priority_map = {rule_prefix: idx for idx, rule_prefix in enumerate(rule_priority)}
        rule_config["RULES"] = sorted(rule_config["RULES"], key=lambda item: rule_priority_map.get(item["PREFIX"], -1))
        for rule in rule_config["RULES"]: 
            sub_df = rule["SUB_DF"]
            if "SUB_RULES" in rule:
                sub_df_list, sub_priority = self._add_tag(sub_df, rule["SUB_RULES"], priority_base)
                tag2priority.update(sub_priority)
                priority_base += len(sub_priority)
                tagged_df_list.extend(sub_df_list)
            elif not sub_df.empty:
                for tag in sub_df["tag"].unique():
                    tag2priority[tag] = priority_base
                priority_base += 1
                tagged_df_list.append(sub_df)

        return tagged_df_list, tag2priority

    def add_tag_for_gpu_df(self, rule_config_path: str):
        rule_config = json.load(open(rule_config_path))
        self._add_gpu_annotation(rule_config.get("IGNORE_GPU_ANNOTATION", None))
        self.gpu_df["tag"] = ""
        tagged_df_list, tag2priority = self._add_tag(self.gpu_df, rule_config)
        for df in tagged_df_list:
            df["tag"] = df["tag"].str.strip(": ")
        self.tag2priority = {tag.strip(": "): priority for tag, priority in tag2priority.items()}
        self.gpu_df = pd.concat(tagged_df_list).sort_values(by="start_ts")

class KernelBreakdownAnalyzer:
    def __init__(self, trace: TorchProfile):
        self.percent_threshold = 0.0
        self.trace = trace
        required_columns = ["tag", "start_ts", "end_ts"]
        assert set(required_columns).issubset(trace.gpu_df.columns), f"columns {required_columns} is required, got {trace.gpu_df.columns}"
        
        self.all_tags: List[str] = trace.gpu_df["tag"].unique()
        self.all_tags = sorted(self.all_tags, key=lambda tag: self.trace.tag2priority[tag])
        self.tag2idx: Dict[str, int] = {}
        self.idx2tag: Dict[int, str] = {}
        self.tag2df: Dict[str, pd.DataFrame] = {}
        for idx, tag in enumerate(self.all_tags):
            self.tag2idx[tag] = idx
            self.idx2tag[idx] = tag
            self.tag2df[tag] = self._merge_intervals(trace.gpu_df[trace.gpu_df["tag"].eq(tag)].copy())
        
        status_df = pd.DataFrame({
            "status": pd.Series(dtype="str"),
            "time": pd.Series(dtype="int"),
        })
        for tag, tag_df in self.tag2df.items():
            value = 1 << self.tag2idx[tag]
            melted_df = tag_df[['start_ts', 'end_ts']].melt(var_name="status", value_name="time").replace(
                {"start_ts": value, "end_ts": -value}
            ).infer_objects()
            status_df = pd.concat([status_df, melted_df]).sort_values(by="time").reset_index(drop=True)

        status_df["running"] = status_df["status"].cumsum()
        status_df["status"] = status_df["running"]
        status_df["next_time"] = status_df["time"].shift(-1)
        self.status_df = status_df[status_df["status"].ge(0)]
        # assert status_df[status_df["status"].lt(0)].empty, f"got {status_df[status_df['status'].lt(0)]} for status < 0"
    
    @staticmethod
    def _merge_intervals(df: Union[pd.DataFrame, List[pd.DataFrame]], is_sorted: bool = False) -> pd.DataFrame:
        """ 
        Merge All the Intervals in a DataFrame,
        Inputs: dataframe in which the intervals are represented by "start_ts" and "end_ts"
        e.g: input:
            start_ts  end_ts
            0      2    10
            1      6    15
            2     20    25
        Returns:
            a new DataFrame in which all the invervals are merged
        e.g: output:
            start_ts  end_ts
            0      2    15
            1      20   25
        """
        if isinstance(df, list):
            df = pd.concat(df)
        df = df[df['end_ts'] >= df['start_ts']]
        if not is_sorted:
            df.sort_values(by="start_ts", inplace=True)
        df.loc[:, "group"] = (df.loc[:, "start_ts"] > df.loc[:, "end_ts"].shift().cummax()).cumsum()
        df = (
            df.groupby("group", as_index=False)
            .agg({"start_ts": "min", "end_ts": "max"})
            .drop(["group"], axis=1)
            .sort_values(by="start_ts")
        )
        return df

    def _find_idle(self) -> pd.DataFrame:
        idle_df = self.status_df[self.status_df["status"].eq(0)].copy().rename(columns={
            "time": "start_ts",
            "next_time": "end_ts"   
        }).drop(["status"], axis=1)
        idle_df["duration"] = idle_df["end_ts"] - idle_df["start_ts"]
        return idle_df
    
    def _find_intersection(self, tag_list: List[str]):
        target_status = sum((1 << self.tag2idx[tag]) for tag in tag_list)
        overlapped_df = self.split_df[self.split_df["status"].eq(target_status)].copy().rename(columns={
            "time": "start_ts",
            "next_time": "end_ts"   
        }).drop(["status"], axis=1)
        overlapped_df["duration"] = overlapped_df["end_ts"] - overlapped_df["start_ts"]
        return overlapped_df

    def _find_substraction(self, target_tag_list: Union[int, List[int]], substracted_tag_list: List[int]):
        if isinstance(target_tag_list, int):
            target_tag_list = [target_tag_list]
        target_status = sum(1 << self.tag2idx[tag] for tag in target_tag_list)
        substracted_status = sum((1 << self.tag2idx[tag]) for tag in substracted_tag_list)
        condition1 = (self.status_df["status"] & target_status) > 0
        condition2 = (self.status_df["status"] & substracted_status) == 0
        substracted_df = self.status_df[condition1 & condition2].copy().rename(columns={
            "time": "start_ts",
            "next_time": "end_ts"   
        }).drop(["status"], axis=1)
        substracted_df["duration"] = substracted_df["end_ts"] - substracted_df["start_ts"]
        return substracted_df

    def _print_results(self, result_df: pd.DataFrame):
        result_df.sort_values(by="duration", ascending=False, inplace=True)
        result_df["duration"] = result_df["duration"] / 1e6 # convert ns to ms
        result_df["percentage"] = result_df["percentage"] * 100.0
        result_df = result_df[result_df["percentage"] > self.percent_threshold]
        result_df.rename(columns={"duration": "duration(ms)", "percentage": "percentage(%)"}, inplace=True)
        print(result_df.to_markdown(index=False, floatfmt=".2f"))

    def analyze(self, verbose: bool = False):
        result_df = pd.DataFrame({
            "tag": pd.Series(dtype="str"),
            "duration":pd.Series(dtype="float"),
            "percentage": pd.Series(dtype="float"),
        })
        high_priority_tags = []
        for tag in self.all_tags:
            tag_df = self._find_substraction([tag], high_priority_tags)
            result_df = pd.concat([
                result_df,
                pd.DataFrame([{
                    "tag": tag,  
                    "duration":tag_df["duration"].sum(), 
                    "percentage": tag_df["duration"].sum() * 1.0 / self.trace.trace_duration_us
                }]),
            ], ignore_index=True)
            high_priority_tags.append(tag)
        idle_df = self._find_idle()
        result_df = pd.concat([
            result_df,
            pd.DataFrame([{
                "tag": "Idle",
                "duration":idle_df["duration"].sum(), 
                "percentage": idle_df["duration"].sum() * 1.0 / self.trace.trace_duration_us
            }]),
        ], ignore_index=True)
        result_df.sort_values(by="duration", ascending=False, inplace=True)
        if verbose:
            self._print_results(result_df)
        return result_df

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze GPU kernels from PyTorch profiler JSON trace files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n%(prog)s --trace-file trace.json --rule-config ./configs/base.json",
    )

    parser.add_argument("--trace-path", help="Path to the PyTorch profiler JSON trace file", type=str, required=True)
    parser.add_argument("--rule-config-path", help="Path to the rule file", type=str, default="./configs/base.json")
    parser.add_argument("--percent-threshold", help="Percentage threshold", type=float, default=0.5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trace = TorchProfile(args.trace_path)
    trace.add_tag_for_gpu_df(args.rule_config_path)

    analyzer = KernelBreakdownAnalyzer(trace)
    analyzer.percent_threshold = args.percent_threshold
    result_df = analyzer.analyze(verbose=True)
    # from IPython import embed; embed()