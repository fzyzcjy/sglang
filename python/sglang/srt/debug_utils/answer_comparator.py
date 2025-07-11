import polars as pl
import argparse


def main(args):
    df_input = pl.concat([
        _read_df_raw(p, category=category, trial_index=i)
        for category, paths in [
            ("baseline", args.baseline_path),
            ("target", args.target_path),
        ]
        for i, p in paths
    ])
    assert all(c in df_input.columns for c in ["category", "trial_index", "prompt_id", "prompt", "answer"])

    df_meta = pl.DataFrame([
        dict(
            prompt_id=prompt_id,
            **_handle_one_prompt(df_input.filter(pl.col("prompt_id") == prompt_id)),
        )
        for prompt_id in sorted(df_input["prompt_id"].to_list())
    ])

    TODO


def _read_df_raw(path: str, category: str, trial_index: int):
    return pl.read_json(path).with_columns(category=pl.lit(category), trial_index=trial_index)


def _handle_one_prompt(df_one_prompt: pl.DataFrame):
    df_one_prompt = df_one_prompt.sort("category", "trial_index")
    assert len(set(df_one_prompt["prompt"])) == 1
    
    return dict(
        TODO=TODO,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str, nargs="+")
    parser.add_argument("--target-path", type=str, nargs="+")
    args = parser.parse_args()
    main(args)
