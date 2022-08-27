
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://www.ulaval.ca/en/" target="_blank">
    <img src="https://ssc.ca/sites/default/files/logo-ulaval-reseaux-sociaux.jpg" alt="Logo" width="200" height="100">
  </a>

  <h3 align="center">LeManchot-VT-Smart</h3>

  <p align="center">
	The VT Smart subsystem of the LeManchot platform provides the required tools for fusion of thermal and visivle modalities. Also it provides the required features to detect the defects and abnormalities using both modalities.
    <br/>
    <br/>
  </p>
</p>

## Experiments

### Case Study 1 : ULAVAL Road Inspection:
- We are going to use coupled thermal and visible images for inspection of pavement roads.
- Since the drone's altitude is static for each data session, we will use the GCP placed at begining of each session to determine the parameters for modality registration.
**OBJECTIVES**: We will use the visible image to segment vegetation and pavement in visible images, then use these blobs to segment thermal images.

### Case Study 2 : Piping Inspection
- We are going to use coupled thermal and visible images for piping inspection.
**OBJECTIVE**: (a) Fusion of thermal and visible images, (b) Distinguish normal and abnormal defects for industrial inspection.

### Case Study 3 : Bridge Inspection
- We are going to use coupled thermal and visible images for bridge inspection.
**OBJECTIVE**: (a) Registration of thermal and visible images, (b) Extract concrete areas from thermal images using segmentation of visible images.


# LeManchot-VT-Smart

<!-- ![Diagram Image Link](./design/method.puml) -->

.... under construction

## Usage

For setting the path for system settings you need to initialize system environment variable: 

``` $ export LEMANCHOT_VT_SETTING_PATH = [ADD YOUR PATH HERE] ```

## Citation

.... the paper will be publish soon!

## Contributors
**Parham Nooralishahi** - parham.nooralishahi@gmail.com | [@phm](https://www.linkedin.com/in/parham-nooralishahi/) <br/>
**Gabriel Ramos** - gabriel.ramos.1@ulaval.ca | [@gabriel](https://www.linkedin.com/in/gramos-ing/) <br/>
**Sandra Pozzer** - sandra.pozzer.1@ulaval.ca | [@sandra](https://www.linkedin.com/in/sandra-pozzer/) <br/>

## Team
**Parham Nooralishahi** is a specialist in embedded and intelligent vision systems and currently is a Ph.D. student at Universite Laval working on developing drone-enabled techniques for the inspection of large and complex industrial components using multi-modal data processing. He is a researcher with a demonstrated history of working in the telecommunication industry and industrial inspection and in-depth expertise in robotics & drones, embedded systems, advanced computer vision and machine learning techniques. He has a Master’s degree in Computer Science (Artificial Intelligence). During his bachelor's degree, he was involved in designing and developing the controlling and monitoring systems for fixed-wing drone for search and rescue purposes. Also, during his Master's degree, he worked extensively on machine learning and computer vision techniques for robotic and soft computing applications.

**Gabriel Ramos** received his Bachelor's degree in Mechanical Engineering (B.Eng.) from Universit\'e Laval, Quebec, Canada in 2017. 
During his bachelor's degree and his early career, he specialised in numerical structural, modal, and thermal simulations, data analysis and mechanical systems design. He is currently a student in the department of Computer Science and Software Engineering at Universit\'e Laval, where he is pursuing his Master's degree in Artificial Intelligence with a focus on computer vision for robotics.

**Sandra Pozzer**

**Fernando Lopez** is a senior scientist with over 12 years of experience in industry and research in infrared (IR) imaging, advanced non-destructive testing and evaluation (NDT&E) of materials, applied heat transfer, and signal processing. After obtaining his Ph.D. in Mechanical Engineering in 2014, he worked as a Postdoctoral Researcher at Universit'e Laval, conducting research projects with various industrial partners, mainly in aerial IR thermography (IRT) inspection, energy efficiency, and robotic IRT for the NDT&E of aerospace components. He has been the recipient of several academic and research awards, including the 2015 CAPES Doctoral Thesis Award in Engineering, 2015 UFSC Honorable Mention Award, Emergent Leaders of the Americas Award from the Ministry of Foreign Affairs and International Trade of Canada and the Best Presentation Award from 7th International Workshop Advances in Signal Processing for NDE of Materials. Dr. Lopez is currently the Chair of the Program Committee of the CREATE NSERC Innovative Program on NDT and a member of the Standard Council Canada ISO/TC 135/SC 8 on Thermographic Testing. His scientific contributions include more than 20 publications in peer-reviewed journals and international conferences. He is currently Director of Research and Development at TORNGATS, leading several R&D initiatives on advanced NDT&E methods.

**Xavier P.V. Maldague** received the B.Sc., M.Sc., and Ph.D. degrees in electrical engineering from Universite Laval, Quebec City, Canada, in 1982, 1984, and 1989, respectively. He has been a Full Professor with the Department of Electrical and Computing Engineering, Universite Laval, Quebec City, Canada, since 1989, where he was the Head of the Department from 2003 to 2008 and 2018. He has trained over 50 graduate students (M.Sc. and Ph.D.) and has more than 300 publications. His research interests include infrared thermography, nondestructive evaluation (NDE) techniques, and vision/digital systems for industrial inspection. He is an Honorary Fellow of the Indian Society of Nondestructive Testing. He is also a Fellow of the Canadian Engineering Institute, the American Society of Nondestructive Testing, and the Alexander von Humbolt Foundation, Germany. He holds the Tier 1 Canada Research Chair in Infrared Vision. He has been the Chair of the Quantitative Infrared Thermography (QIRT) Council since 2004.

## Contact
Parham Nooralishahi - parham.nooralishahi@gmail.com | [@phm](https://www.linkedin.com/in/parham-nooralishahi/) <br/>

## Acknowledgements
This research is supported by the Canada Research Chair in Multi-polar Infrared Vision (MiViM), the Natural Sciences, and Engineering Research Council (NSERC) of Canada through a discovery grant and by the "oN Duty!" NSERC Collaborative Research and Training Experience (CREATE) Program. Special thanks to TORNGATS company for their support in testing and manufacturing of the parts.
