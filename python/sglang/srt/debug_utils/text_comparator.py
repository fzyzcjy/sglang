import argparse
import json

import polars as pl


def main(args):
    df_input = _compute_df_input(args)
    assert all(
        c in df_input.columns
        for c in ["category", "trial_index", "prompt_id", "prompt", "output", "correct"]
    )

    df_meta = _compute_df_meta(df_input)

    df_correctness_per_trial = df_meta.group_by(
        "category", "trial_index", maintain_order=True
    ).agg(pl.col("correct").mean())
    df_correctness_delta = (
        df_meta.group_by("correctness_delta").count().sort("correctness_delta")
    )

    print("====== Correctness per trial ======")
    with pl.Config(fmt_str_lengths=10000, tbl_cols=-1, tbl_rows=-1):
        print(df_correctness_per_trial)

    print("====== Correctness Delta (-1.0 means all-right becomes all-wrong) ======")
    with pl.Config(fmt_str_lengths=10000, tbl_cols=-1, tbl_rows=-1):
        print(df_correctness_delta)

    for name, df in [
        ("Good->Bad", df_meta.filter(pl.col("correctness_delta") < 0)),
        ("Bad->Good", df_meta.filter(pl.col("correctness_delta") > 0)),
    ]:
        print(f"====== Concrete Examples: {name} ======")
        with pl.Config(fmt_str_lengths=10000, tbl_cols=-1, tbl_rows=-1):
            print(df)


def _compute_df_input(args):
    return pl.concat(
        [
            _read_df_raw(p, category=category, trial_index=i)
            for category, paths in [
                ("baseline", args.baseline_path),
                ("target", args.target_path),
            ]
            for i, p in enumerate(paths)
        ]
    )


def _read_df_raw(path: str, category: str, trial_index: int):
    return pl.read_json(path).with_columns(
        category=pl.lit(category), trial_index=trial_index
    )


def _compute_df_meta(df_input: pl.DataFrame):
    df_meta = pl.DataFrame(
        [
            _handle_one_prompt(df_input.filter(pl.col("prompt_id") == prompt_id))
            for prompt_id in sorted(set(df_input["prompt_id"].to_list()))
        ]
    )
    df_meta = df_meta.with_columns(
        correctness_delta=pl.col("correctness_target") - pl.col("correctness_baseline"),
    )
    df_meta = df_meta.sort("correctness_delta", "output_same_prefix_len")
    return df_meta


def _handle_one_prompt(df_one_prompt: pl.DataFrame):
    df_one_prompt = df_one_prompt.sort("category", "trial_index")
    assert len(set(df_one_prompt["prompt"])) == 1

    df_baseline = df_one_prompt.filter(pl.col("category") == "baseline")
    df_target = df_one_prompt.filter(pl.col("category") == "target")

    outputs_baseline = df_baseline["output"].to_list()
    outputs_target = df_baseline["target"].to_list()

    output_same_prefix_len = max(
        [
            _compute_str_prefix_len(output_baseline, output_target)
            for output_baseline in outputs_baseline
            for output_target in outputs_target
        ]
    )

    return dict(
        prompt_id=df_one_prompt[0, "prompt_id"],
        correctness_baseline=df_baseline["correct"].mean(),
        correctness_target=df_target["correct"].mean(),
        output_same_prefix_len=output_same_prefix_len,
        prompt_escaped=json.dumps(df_one_prompt[0, "prompt"]),
        outputs_baseline=outputs_baseline,
        outputs_target=outputs_target,
    )


def _compute_str_prefix_len(a: str, b: str) -> int:
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    return min_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str, nargs="+")
    parser.add_argument("--target-path", type=str, nargs="+")
    args = parser.parse_args()
    main(args)
