<h2 align="center"><a href="https://arxiv.org/abs/2506.13737">ExtendAttack: Attacking Servers of LRMs via Extending Reasoning</a></h2>

<p align="center">
  <img src="imgs/ExtendAttack.png" alt="ExtendAttack Overview" width="1000"/>
</p>

## Quick Start
To evaluate ExtendAttack, you can use the following setup.

1. Create an environment and install dependencies
    ```
    conda create -n extendattack python=3.11 -y
    conda activate extendattack
    pip install -r requirements.txt
    ```

2. Set the API configuration
    ```
    export API_KEY="your_api_key"
    # Optional: set this when using an OpenAI-compatible proxy or provider.
    export API_BASE_URL="https://your-openai-compatible-endpoint/v1"

    # Optional: override the judge model used by Matheval.py.
    export JUDGE_MODEL="gpt-4o-mini"
    ```

3. Inference
    ```
    # for AIME2024
    python src/Math.py --model o3-mini --ratio 0.1 --max-parallel 30 --n 1 --dataset aime2024 --method ExtendAttack

    # for AIME2025
    python src/Math.py --model o3-mini --ratio 0.1 --max-parallel 30 --n 1 --dataset aime2025 --method ExtendAttack

    # for HumanEval
    python src/humaneval.py --model o3-mini --ratio 0.5 --max-parallel 30 --n 1 --dataset humaneval --method ExtendAttack

    # for BigCodeBench-Complete
    python src/bigcodebench.py --model o3-mini --ratio 0.2 --max-parallel 30 --n 1 --dataset bigcodebench --method ExtendAttack
    ```

4. Evaluate

    For BigCodeBench-Complete, please refer to [bigcodebench](https://github.com/bigcode-project/bigcodebench).

    ```
    # for AIME2024
    python src/Matheval.py --model o3-mini --ratio 0.1 --n 1 --dataset aime2024 --method ExtendAttack

    # for AIME2025
    python src/Matheval.py --model o3-mini --ratio 0.1 --n 1 --dataset aime2025 --method ExtendAttack

    # for HumanEval
    python src/humanevaleval.py --model o3-mini --ratio 0.5 --n 1 --dataset humaneval --method ExtendAttack
    ```

For a quick smoke test before launching a full run, you can append `--limit 1` to any inference command.

## Citations

If you find this repository helpful, please cite our paper.

```
@misc{zhu2025extendattackattackingserverslrms,
      title={ExtendAttack: Attacking Servers of LRMs via Extending Reasoning},
      author={Zhenhao Zhu and Yue Liu and Zhiwei Xu and Yingwei Ma and Hongcheng Gao and Nuo Chen and Yanpei Guo and Wenjie Qu and Huiying Xu and Zifeng Kang and Xinzhong Zhu and Jiaheng Zhang},
      year={2025},
      eprint={2506.13737},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.13737},
}
```
