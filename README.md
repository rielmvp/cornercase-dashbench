# DashBench: Evaluating Video LLMs' Understanding of Corner Cases in Autonomous Driving

<div align="center">
  <p>
    Gabriel Manalu, Minh Son Hoang, Hyeontaek Hwang, Jaesung Lee, Hyojin Choi, Tran Tuan Phong, Seon Kwon Kim,
    Saad Wazir, Daeyoung Kim
  </p>
  <p>School of Computing, KAIST, Daejeon, Korea</p>
  <h3>BMVC 2026</h3>

[![Paper](https://img.shields.io/badge/Paper-arXiv_2611.16767-b31b1b.svg)](https://arxiv.org/abs/2611.16767)
[![Project Page](https://img.shields.io/badge/Project_Page-cornercase--dashbench.github.io-blue.svg)](https://cornercase-dashbench.github.io)
[![Code License](https://img.shields.io/badge/Code_License-MIT-green.svg)](LICENSE)

</div>

---

<p align="center">
  <img src="./figures/fig1_overview.jpg" alt="DashBench overview" width="85%"/>
</p>

## Abstract

*Natural language explanations promised by modern Video Large Language Models (Video LLMs) offer autonomous driving systems a path toward detailed scene description and interpretable decision making on the road. While recent benchmarks have begun assessing these models on driving scenes, they mainly target routine traffic conditions and general comprehension tasks based on existing driving datasets and therefore fail to evaluate model performance on the rarest and most safety-critical driving events, also known as "corner cases". To provide a comprehensive evaluation of Video LLMs on precisely these scenarios, we introduce DashBench, a benchmark testing 25 Video LLMs on 1,709 carefully curated driving videos spanning not only collision-related corner cases but also rare non-collision ones. Through extensive prompts and open-ended question answering, we expose a critical capability gap: all evaluated models perform substantially below the threshold required for safety-critical driving applications, frequently misjudging or missing dangerous events on the road ahead and hallucinating non-existent agents or actions in their explanations.*

## Overview

| | |
|---|---|
| Clips | **1,709** ego-view dashcam clips (MM-AU 1,395, RoadSocial 314) |
| Categories | **88**: 46 collision and 42 non-collision corner cases (890 / 819 clips) |
| Input | 16 frames per clip, uniformly sampled across the clip (`full`) or inside the annotated corner-case window (`ccw`, 823 clips) |
| Prompts | 5 open-ended prompts (Q1–Q5) per clip, 8,545 clip–prompt pairs |
| Judge | DeepSeek-V3 compares each response with the ground truth: same core cause, effect and timing of the danger? Yes/No (Cohen's κ = 0.96 with humans) |
| Models evaluated | 25 Video LLMs: 2 closed-source, 16 open-source, 7 driving specialists |

Each clip was manually verified to contain a visually identifiable, self-contained corner case, annotated with keyframes, and given a one-sentence ground-truth description drafted with OpenAI o3 and refined to consensus by three annotators.

<p align="center">
  <img src="./figures/fig2_pipeline.jpg" alt="DashBench construction pipeline" width="85%"/>
</p>

## Results

Accuracy (%) of the top 10 of 25 Video LLMs on all 1,709 clips, ranked by Avg. Q4R is the reasoning of the JSON prompt, Q4TF its binary detection answer, and Avg the mean over the prompts a model was evaluated on. The full table of 25 models is in the [paper](https://arxiv.org/abs/2611.16767).

| Model | Q1 | Q2 | Q3 | Q4R | Q5 | Avg | Q4TF |
|---|---|---|---|---|---|---|---|
| GPT-5 | 19.95 | 36.92 | 31.19 | 46.52 | 39.15 | 34.75 | 97.83 |
| Gemini-3-Flash | 32.36 | 35.28 | 29.84 | 39.58 | 34.99 | 34.41 | 94.85 |
| Qwen3-VL (235B) | 18.72 | 20.19 | 16.79 | 25.92 | 21.71 | 20.67 | 72.03 |
| Qwen3-VL (8B) | 18.90 | 15.51 | 14.45 | 22.00 | 21.53 | 18.48 | 74.96 |
| Qwen3-VL (32B) | 17.20 | 18.90 | 12.87 | 18.78 | 18.31 | 17.21 | 71.04 |
| Ovis2.5 (9B) | 19.68 | 15.41 | 7.77 | 18.72 | 20.55 | 16.43 | 64.47 |
| InternVL3.5 (241B) | 19.31 | 12.81 | 11.47 | 19.13 | 17.61 | 16.07 | 76.36 |
| InternVL3.5 (38B) | 12.64 | 13.22 | 8.66 | 16.44 | 12.93 | 12.78 | 77.65 |
| Tarsier2 (7B) | 13.28 | 13.11 | 11.70 | – | 12.35 | 12.61 | – |
| MiniCPM4.5 (8B) | 13.34 | 11.18 | 6.38 | 17.27 | 13.53 | 12.34 | 87.53 |

## Citation

```bibtex
@misc{manalu2026evaluatingvideollmsunderstanding,
      title={Evaluating Video LLMs' Understanding of Corner Cases in Autonomous Driving}, 
      author={Gabriel Manalu and Minh Son Hoang and Hyeontaek Hwang and Jaesung Lee and Hyojin Choi and Tran Tuan Phong and Seon Kwon Kim and Saad Wazir and Daeyoung Kim},
      year={2026},
      eprint={2611.16767},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.16767}, 
}
```

## License

The code in this repository is released under the [MIT License](LICENSE).

DashBench clips are derived from [MM-AU](https://github.com/jeffreychou777/LOTVS-MM-AU) and [RoadSocial](https://roadsocial.github.io/). Please consider their licenses and usage requirements.
