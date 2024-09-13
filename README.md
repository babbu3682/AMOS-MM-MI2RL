# Revisiting M3D is all you need
<font size=3><div align='center' > [**Model**](https://huggingface.co/babbu3682/AMOS-MM-MI2RL-Solution1) | [**Code**](https://github.com/babbu3682/AMOS-MM-MI2RL)</div></font>
This code is the 3rd place solution for the MICCAI 2024 AMOS-MM Challenge. It is based on the M3D model and utilizes the LLaMA-3.1-Instruct-7B.

# Novelty:
* We divided the body into three regions: chest, abdomen, and pelvis. This allowed us to focus more effectively on these areas and further improve performance through text prompt engineering.
* The introduction of LLaMA-3.1-8B-Instruct significantly boosted MRG and VQA performance.
* Additionally, the use of an error note strategy led to further improvements in the final results.

# Acknowledgement
We appreciate open-source projects including: 
[M3D](https://github.com/BAAI-DCAI/M3D),
[LLaVA](https://github.com/haotian-liu/LLaVA),
[LLaMA](https://github.com/meta-llama/llama3),
