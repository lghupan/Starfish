# Starfish
Code Repo for ["Starfish: Resilient Image Compression for AIoT Cameras"](https://panhu.me/pdf/2020/Starfish.pdf)

## 2021/06/11 update
We've updated the JPEG configuration to avoid confusion about the performance gain of Starfish. Please check `demo_size_quality_optimized_jpeg.ipynb` and chapter 3 of `Dissertation_augmented_small.pdf` for a more comprehensive understanding of the Starfish work.

## How to replicate
0. Hardware requirements: 32GB RAM, 128G Disk and Nvidia GPU with 16GB RAM. Optional: K210 board which could be obtain from:
https://www.seeedstudio.com/Sipeed-M1-dock-suit-M1-dock-2-4-inch-LCD-OV2640-K210-Dev-Board-1st-RV64-AI-board-for-Edge-Computing.html
1. Download Dataset and trained models (~8GB in total, md5=`db631866a594529141d5513a9424d215`) from https://drive.google.com/file/d/172KaUzWDaSdYPNFMV2CP9UuZN5evERKY/view?usp=sharing
2. Create a virtual environment with Python 3.7 and install requirments `pip install -r requirements.txt`
3. Start Jupyter notebook from current directory: `jupyter notebook`
4. From the Web interface of Jupyter notebook, select `demo_size_quality.ipynb`. Click "Cell"-> "Run All" from the main memu to run benchmark on the size and quality of StarFish. The figure will be saved to the `Outputs` folder, which should be similar to Figure 13 (left) in the paper.
5. In `demo_size_quality.ipynb`, modify `train_target` and `eval_target` to 1 and 2, then re-run all cells to generate middle and right sub-figure of Figure 13.
6. Similarly, run `demo_resiliency.ipynb` to replicate Figure 11 and 18 in the paper, and `LoRaSim/LoRaSim.ipynb` to replicate Figure 19.
7. *Optional* steps for compiling and running the model on K210 board:
-- Compile the model: create another virtual environment and install Tensorflow 1.15. Set `check = True` in Section 1.2 to compile the model.
-- Following the instructions on https://maixpy.sipeed.com/maixpy/en/how_to_read.html to setup the dev environment.
-- Flash the firmware provided in `K210` folder.
-- From the Maxipy IDE, run `K210/time.maixpy` to obtain output on K210 board.

## Explain of  folder and files
- Dataset: location of raw and cooked datasets.
- K210: compiler, firmware and Maxipy for the K210 board.
- LoRaSim: simulation files for LoRa, based on simulator from https://www.lancaster.ac.uk/scc/sites/lora/lorasim.html
- Output: figures and output results
- SavedModel: pretained and trained DNN models
- demo_resiliency.ipynb: demostrate the resiliency of StarFish (as Figure 11 and 18)
- demo_size_quality.ipynb: demostrate the size-quality trade-off of StarFish (as Figure 13)
- net.py: definition and helper functions to create and evaluate DNN
- utils.py: helper functions for matrix operation

## Bibtex
    @inproceedings{hu2020starfish,
      title={Starfish: resilient image compression for AIoT cameras},
      author={Hu, Pan and Im, Junha and Asgar, Zain and Katti, Sachin},
      booktitle={Proceedings of the 18th Conference on Embedded Networked Sensor Systems},
      pages={395--408},
      year={2020}
    }
