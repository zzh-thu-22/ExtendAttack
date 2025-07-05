# ExtendAttack: Attacking Servers of LRMs via Extending Reasoning

This is the official implementation of the paper: *ExtendAttack: Attacking Servers of LRMs via Extending Reasoning*.

<p align="center">
  <img src="imgs/ExtendAttack.png" alt="ExtendAttack Overview" width="1000"/>
</p>

## 🚀 Quick Start
To evaluate ExtendAttack, you should run the following codes.

1. Set the API keys
    ```
    export API_KEY="your_api_key"
    ```

2. Inference
    ```
    # for AIME2024
    python src/Math.py --model o3-mini --ratio 0.1 --max-parallel 30 --n 1 --dataset aime2024 --method ExtendAttack

    # for AIME2025
    python src/Math.py --model o3-mini --ratio 0.1 --max-parallel 30 --n 1 --dataset aime2025 --method ExtendAttack

    # for HumanEval
    python src/humaneval.py --model o3-mini --ratio 0.5 --max-parallel 30 --n 1 --dataset humaneval --method ExtendAttack

    # for BigcodeBench-Complete
    python src/bigcodebench.py --model o3-mini --ratio 0.2 --max-parallel 30 --dataset bigcodebench --method ExtendAttack
    ```

3. Evaluate

    For BigCodeBench-Complete, please refer to [bigcodebench](https://github.com/bigcode-project/bigcodebench).

    ```
    # for AIME2024
    python src/Matheval.py --model o3-mini --ratio 0.1 --n 1 --dataset aime2024 --method ExtendAttack

    # for AIME2025
    python src/Matheval.py --model o3-mini --ratio 0.1 --n 1 --dataset aime2025 --method ExtendAttack

    # for HumanEval
    python src/humanevaleval.py --model o3-mini --ratio 0.5 --n 1 --dataset humaneval --method ExtendAttack
    ```
    


## 📜 Citations

If you find this repository helpful, please cite our paper.

```
@article{zhu2025extendattackattackingserverslrms,
      title={ExtendAttack: Attacking Servers of LRMs via Extending Reasoning}, 
      author={Zhenhao Zhu and Yue Liu and Yingwei Ma and Hongcheng Gao and Nuo Chen and Yanpei Guo and Wenjie Qu and Huiying Xu and Xinzhong Zhu and Jiaheng Zhang},
      year={2025},
      eprint={2506.13737},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.13737}, 
}
```