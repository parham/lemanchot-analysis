<div align="center">
  <img src="https://github.com/parham/parham.github.io/blob/main/assets/img/favicon.png"/ width="200">
</div>

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://www.ulaval.ca/en/" target="_blank">
    <img src="https://ssc.ca/sites/default/files/logo-ulaval-reseaux-sociaux.jpg" alt="Logo" width="200" height="100">
  </a>

  <h3 align="center">LeManchot-Analysis</h3>

  <p align="center">
	The Analysis subsystem of the LeManchot platform provides the required tools for fusion of thermal and visivle modalities. Also it provides the required features to detect the defects and abnormalities using both modalities.
    <br/>
    <br/>
  </p>
</p>

# LeManchot-Analysis
<p align="justify"> Remote thermographic inspection of industrial and construction components is a relatively new field of research with growing interest among researchers and companies, especially in the light of recent advancements in Unmanned Aerial Vehicles (UAVs). Due to the unique capabilities of drones to carry a wide range of sensors as payload and facilitate data acquisition through their maneuver abilities, collecting synchronized multi-modal data has become a possibility in terms of cost and time. In addition, using multiple sensors can significantly enhance the analysis result and provide a more accurate assessment of the components' operability and structural integrity. Complementary modalities oblige companies to build automated process pipelines that can avoid data misinterpretation and deliver more precise and comprehensive evaluations, which is one of the objectives of NDT4.0. In this paper, we investigate the applications of texture-based segmentation of visible modality to enhance thermal analysis in a drone-based multi-modal inspection. The proposed process pipeline texturally segments the visible images and uses the result to improve the detection and characterization of possible abnormalities in thermal images. Moreover, we introduced four case studies with different process pipelines and datasets to demonstrate the presented approach's benefits in various industrial applications.</p>

**Keywords** `Multi-Modal Data Processing`  `Unmanned Aerial Vehicle`   `Texture Segmentation`   `Remote Inspection`   `Thermography`   `Thermal Image Segmentation`

## Citation

```
@article{nooralishahi2022texture,
  title={Texture analysis to enhance drone-based multi-modal inspection of structures},
  author={Nooralishahi, Parham and Ramos, Gabriel and Pozzer, Sandra and Ibarra-Castanedo, Clemente and Lopez, Fernando and Maldague, Xavier PV},
  journal={Drones},
  volume={6},
  number={12},
  pages={407},
  year={2022},
  publisher={MDPI}
}
```

## Links

- **Analysis & all Trainings**: <a href="https://www.comet.com/parham/comparaWtive-analysis/view/OIZqWwU2dPR1kOhWH9268msAC/experiments">comet.ml repo</a>


## Usage

For setting the path for system settings you need to initialize system environment variable: 

``` $ export LEMANCHOT_VT_SETTING_PATH = [ADD YOUR PATH HERE] ```
For the system to be able to load the experiment configuration, you need to initialize configuration dir path:
``` $ export LEMANCHOT_VT_CONFIG_DIR = [CONFIG DIR] ```

## Use Cases

<p align="justify"> This section includes four use cases of employing coupled thermal and visible cameras aiming toward the enhancement of post-analysis during a drone-based automated process pipeline in different inspection scenarios in different industries. The **Use Case 1** explains the benefits of fusing visible images with thermal images to enhance the defect detection process. **Use Case 2** explains the multi-modal approach for abnormality classification in piping inspection. Employing visible images to extract the region of interest in thermal images to enhance the drone-based thermographic inspection of roads is described in **Use Case 3**. Finally, a drone-based inspection of concrete bridges using coupled thermal and visible cameras is investigated in **Use Case 4**. </p>

### Case Study 1 : Enhance Visual Inspection of Roads using Coupled Thermal and Visible Cameras 

<p align="justify"> One of the applications for coupled thermal and visible sensors is to enhance defect detection in visual inspection using thermal images in case of illumination or contrast issues. This section presents a process pipeline for automatic crack detection using coupled thermal and visible images. The objective is to demonstrate the effect of the Thermal-Visible image fusion on crack detection in typical- and worst-case scenarios. The worst-case scenario occurs when shadows, low illumination, or low contrast disrupt the detection process. </p>

<p align="center">
  <img src="resources/use_case_1.png" width="500" title="Dataset for Use Case 1">
</p>

<p align="center">
  <img src="resources/result_uc_1_1.png" width="500" title="Result for Use Case 1">
</p>

<p align="center">
  <img src="resources/result_uc_1_2.png" width="500" title="Result for Use Case 1">
</p>

### Case Study 2 : Abnormality Classification using Coupled Thermal and Visible Images

<p align="justify"> Another area in which coupled thermal and visible images can be beneficial is remote inspection when physical access is limited. In such scenarios, comprehensive information in different modalities is needed to avoid data misinterpretation. In the case of thermographic inspection, the abnormalities are recognizable in thermal images, and several methods exist that can semi-automate the detection process. However, distinguishing between surface and subsurface defects is hard or impossible with only thermal information in an automated process pipeline. To address this challenge, coupled thermal and visible images can be employed to enhance the classification process. In this use case, thermal and visible images are used to classify detected defects into surface and subsurface abnormalities using texture analysis. </p>

<p align="center">
  <img src="resources/result_uc_2_1.png" width="500" title="Result for Use Case 2">
</p>

### Case Study 3 : Enhance the Analysis of Drone-based Road Inspection using Coupled Thermal and Visible Images

In this use case, the use of visible images for helping to extract the region of interest in thermal images is investigated comprehensively for drone-based inspection of road pavement. 

<p align="center">
  <img src="resources/use_case_2.png" width="500" title="Dataset for Use Case 3">
</p>

<p align="center">
  <img src="resources/result_uc_3_1.png" width="500" title="Result for Use Case 3">
</p>

<p align="center">
  <img src="resources/result_uc_3_2.png" width="500" title="Result for Use Case 3">
</p>

### Case Study 4 : nhance the Analysis of Bridge Inspection using Coupled Thermal and Visible Images

This use case investigates the use of coupled thermal and visible cameras to enhance the drone-based thermographic inspection of concrete bridges.

<p align="center">
  <img src="resources/use_case_3.png" width="500" title="Dataset for Use Case 4">
</p>

<p align="center">
  <img src="resources/result_uc_4_1.png" width="500" title="Result for Use Case 4">
</p>

<p align="center">
  <img src="resources/result_uc_4_2.png" width="500" title="Result for Use Case 4">
</p>

## Contributors

**Parham Nooralishahi** - parham.nooralishahi@gmail.com | [@phm](https://www.linkedin.com/in/parham-nooralishahi/) <br/>
**Gabriel Ramos** - gabriel.ramos.1@ulaval.ca | [@gabriel](https://www.linkedin.com/in/gramos-ing/) <br/>
**Sandra Pozzer** - sandra.pozzer.1@ulaval.ca | [@sandra](https://www.linkedin.com/in/sandra-pozzer/) <br/>

## Team

<p align="justify"> <b>Parham Nooralishahi</b> is a specialist in embedded and intelligent vision systems and currently is a Ph.D. student at Universite Laval working on developing drone-enabled techniques for the inspection of large and complex industrial components using multi-modal data processing. He is a researcher with a demonstrated history of working in the telecommunication industry and industrial inspection and in-depth expertise in robotics & drones, embedded systems, advanced computer vision and machine learning techniques. He has a Master’s degree in Computer Science (Artificial Intelligence). During his bachelor's degree, he was involved in designing and developing the controlling and monitoring systems for fixed-wing drone for search and rescue purposes. Also, during his Master's degree, he worked extensively on machine learning and computer vision techniques for robotic and soft computing applications. </p>

<p align="justify"> <b>Gabriel Ramos</b> received his Bachelor's degree in Mechanical Engineering (B.Eng.) from Universit\'e Laval, Quebec, Canada in 2017. 
During his bachelor's degree and his early career, he specialised in numerical structural, modal, and thermal simulations, data analysis and mechanical systems design. He is currently a student in the department of Computer Science and Software Engineering at Universit\'e Laval, where he is pursuing his Master's degree in Artificial Intelligence with a focus on computer vision for robotics.</p>

<p align="justify"> <b>Sandra Pozzer</b> received his Bachelor's degree in Civil Engineering (B.Eng.) from the University of Passo Fundo, Brazil (2016). During his bachelor's degree, she specialized in transportation Infrastructure, including one year of applied studies at the Università Degli Studi di Padova, Italy (2013-2014). During her Master's studies, she studied infrared thermography applied to the inspection of concrete structures at the University of Passo Fundo, Brazil ( 2020), including one term of applied research at Lakehead University, Ontario, Canada in 2019. Currently, she is a Ph.D. candidate at Laval Université, Quebec, Canadá, exploring the subjects of infrared thermography, concrete infrastructure, and computer vision applied to civil infrastructure. She has professional experience in the fields of concrete structures, topography, transportation, and infrastructure projects.</p>

<p align="justify"> <b>Xavier P.V. Maldague</b> received the B.Sc., M.Sc., and Ph.D. degrees in electrical engineering from Universite Laval, Quebec City, Canada, in 1982, 1984, and 1989, respectively. He has been a Full Professor with the Department of Electrical and Computing Engineering, Universite Laval, Quebec City, Canada, since 1989, where he was the Head of the Department from 2003 to 2008 and 2018. He has trained over 50 graduate students (M.Sc. and Ph.D.) and has more than 300 publications. His research interests include infrared thermography, nondestructive evaluation (NDE) techniques, and vision/digital systems for industrial inspection. He is an Honorary Fellow of the Indian Society of Nondestructive Testing. He is also a Fellow of the Canadian Engineering Institute, the American Society of Nondestructive Testing, and the Alexander von Humbolt Foundation, Germany. He holds the Tier 1 Canada Research Chair in Infrared Vision. He has been the Chair of the Quantitative Infrared Thermography (QIRT) Council since 2004.</p>

## Contact
Parham Nooralishahi - parham.nooralishahi@gmail.com | [@phm](https://www.linkedin.com/in/parham-nooralishahi/) <br/>

## Acknowledgements
<p align="justify"> We acknowledge the support of the Natural Sciences and Engineering Council of Canada (NSERC), CREATE-oN DuTy Program [funding reference number 496439-2017], DG program, the Canada Research Chair in Multipolar Infrared Vision (MIVIM), and the Canada Foundation for Innovation. Special thanks to TORNGATS company for providing the required equipment and support for performing the experiments. Also, we are grateful to Montmorency Forest of Universit\'e Laval for the kind authorization received to use their experimental road.</p>
