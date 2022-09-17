# FENeRF + StyleCLIP + e4e
## Procedures
### Reference Generation
1. Generate FENeRF samples with random & known camera positions (yaw).
2. Conduct StyleGAN inversion using e4e.

### Editing
3. Conducti editing using StyleCLIP.
4. Extract segmentation map from edted images for FENeRF inversion (editing).
5. Conduct FENeRF editing. There are two versions.
    1. *Original (Limited) FENeRF inversion*: original appearance code + edited segmap
    2. *Maximally assisted with StyleCLIP*: edited image + edited segmap

## How to run
1. Fill out the configurations in `run.sh`
    * **N_SAMPLES**: Number of samples to generate.
    * **OUTPUT_DIR**: A directory to save all the results
    * **FENERF_GENERATOR_PATH**: path of `315000_generator.pth`. Download the files below and put them in the same directory
        * [31500_generator.pth](https://drive.google.com/file/d/18LcYZivCQrrHnGDpu2RUyjUlPA4XFWzf/view?usp=sharing)
        * [315000_ema.pth](https://drive.google.com/file/d/1f605fMk1Wqj-Swq1NhiabiwkF98xDxEl/view?usp=sharing)
        * [315000_ema2.pth](https://drive.google.com/file/d/1bOBWr6ZIbPWJQ8doxF6F2wznGKlRFbDE/view?usp=sharing)
    * **EE_PATH**: path of e4e weights. Download from [here](https://drive.google.com/file/d/17aIovfai0O-Gi5Zt60Sn4mh23bzmNOvl/view?usp=sharing).
    * Note: StyleCLIP model is either automatically downloaded if it's pretraiend by authors, and manually trained models should be saved in `StyleCLIP/mapper/pretrained`. (**TODO**: Structuralize it)

2. Run `bash run.sh ${DEVICE} ${EDIT_TEXT} ${GEN_ORIG} ${USE_"ORIGINAL METHOD}`
    * **DEVICE**: Device num to run
    * **EDIT_TEXT**: Editing text prompt
    * **GEN_ORIG**: if `true`, conduct 1, 2 step. Else, skip 1,2.
    * **USE_ORIGINAL_METHOD**: if `true`, edit via 5.1. Else: eidt via 5.2