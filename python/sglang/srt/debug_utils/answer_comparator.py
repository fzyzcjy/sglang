import polars as pl
import argparse


def main(args):
    df_input = _compute_df_input(args)
    assert all(c in df_input.columns for c in ["category", "trial_index", "prompt_id", "prompt", "answer", "correct"])

    df_meta = _compute_df_meta(df_input)

    df_correctness_delta = df_meta.group_by("correctness_delta").count().sort("correctness_delta")

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
    return pl.concat([
        _read_df_raw(p, category=category, trial_index=i)
        for category, paths in [
            ("baseline", args.baseline_path),
            ("target", args.target_path),
        ]
        for i, p in paths
    ])


def _read_df_raw(path: str, category: str, trial_index: int):
    return pl.read_json(path).with_columns(category=pl.lit(category), trial_index=trial_index)


def _compute_df_meta(df_input: pl.DataFrame):
    df_meta = pl.DataFrame([
        _handle_one_prompt(df_input.filter(pl.col("prompt_id") == prompt_id))
        for prompt_id in sorted(df_input["prompt_id"].to_list())
    ])
    df_meta = df_meta.with_columns(
        correctness_delta=pl.col("correctness_target") - pl.col("correctness_baseline"),
    )
    df_meta = df_meta.sort("correctness_delta", "answer_same_prefix_len")
    return df_meta


def _handle_one_prompt(df_one_prompt: pl.DataFrame):
    df_one_prompt = df_one_prompt.sort("category", "trial_index")
    assert len(set(df_one_prompt["prompt"])) == 1

    df_baseline = TODO
    df_target = TODO

    answers_baseline = df_baseline["answer"].to_list()
    answers_target = df_baseline["target"].to_list()

    answer_same_prefix_len = max([
        _compute_str_prefix_len(answer_baseline, answer_target)
        for answer_baseline in answers_baseline
        for answer_target in answers_target
    ])

    return dict(
        prompt_id=df_one_prompt[0, "prompt_id"],
        correctness_baseline=df_baseline["correct"].mean(),
        correctness_target=df_target["correct"].mean(),
        answer_same_prefix_len=answer_same_prefix_len,
        prompt=df_one_prompt[0, "prompt"],
        answers_baseline=answers_baseline,
        answers_target=answers_target,
    )


def _compute_str_prefix_len(a: str, b: str) -> int:
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    return min_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str, nargs="+")
    parser.add_argument("--target-path", type=str, nargs="+")
    args = parser.parse_args()
    main(args)
