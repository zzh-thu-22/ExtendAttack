# ExtendAttack: Attacking Servers of LRMs via Extending Reasoning

This is the official implementation of the paper: ExtendAttack: Attacking Servers of LRMs via Extending Reasoning.

<figure>
  <img src="imgs/ExtendAttack.png" alt="ExtendAttack Overview" width="1200"/>
</figure>

## 🚀 Quick Start
To evaluate ExtendAttack, you should run the following codes.

1. Set the API keys
    ```
    export API_KEY="your_api_key"
    ```

2. Inference
    ```
    # for AIME2024
    python Math.py --model o3-mini --ratio 0.1 --max-parallel 30 --n 1 --dataset aime2024 --method ExtendAttack

    # for AIME2025
    python Math.py --model o3-mini --ratio 0.1 --max-parallel 30 --n 1 --dataset aime2025 --method ExtendAttack

    # for HumanEval
    python humaneval.py --model o3-mini --ratio 0.1 --max-parallel 30 --n 1 --dataset humaneval --method ExtendAttack

    # for BigcodeBench-Complete
    python bigcodebench.py --model o3-mini --ratio 0.1 --max-parallel 30 --dataset bigcodebench --method ExtendAttack
    ```

3. Evaluate
    ```
    # for AIME2024
    python Matheval.py --model o3-mini --ratio 0.1 --dataset aime2024 --method ExtendAttack

    # for AIME2025
    python Matheval.py --model o3-mini --ratio 0.1 --dataset aime2025 --method ExtendAttack

    # for HumanEval
    python humanevaleval.py --model o3-mini --ratio 0.1 --dataset humaneval --method ExtendAttack
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