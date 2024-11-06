# FMatPix - A Free-Fast-NoDep-Offline Formula Ocr Tool
A free portable Formula Ocr tool supporting latex and mathml based on Texify.<br>
You can use this on your own PC with just CPU and no need for Internet.

# What's the advantage of this?
It's free,portable,offline,fast,accurate.<br>
免费、便携、离线、快速、准确 <br>
So don't hesitate share this with your friends who suffer from paid、user-unfriendly、slow OCR tools.<br>

# How to get the .exe?
从这个网址选择版本下载，注意版本说明里提供了迅雷网盘链接：https://github.com/ai25395/FMatPix/releases <br>
Download the software FMatPix.exe from the release page: https://github.com/ai25395/FMatPix/releases <br>
### Don't forget to give me a star it's very important for me！别忘了给我点个星，这对我真的很重要！<br>

# Version features
## v1.6 v1.7  Page Ocr
<img src='https://github.com/user-attachments/assets/7966e020-7296-45df-9ac9-c244d2cf5f96' width = '750px' height='450px'> <br>
## v1.5 v1.4   Batch Ocr & New History
<img src='https://github.com/user-attachments/assets/019a8292-375f-4386-a426-ac52be159359' width = '750px' height='400px'> <br>

# How to use the software?
1、Alt+Q to snip(no matter in single/batch/page mode) and then the recognition will start.<br>
2、Click 'Copy Latex' or 'Copy MathMl' to copy the recognition result.<br>
3、Double click  'Copy Latex' or 'Copy MathMl' to enable result-auto-copy mode.<br>
4、'History' shows the history snipped pictures of formulas.Click the picture to copy the result.<br>
5、Double click 'Batch Ocr' to enable batch-ocr mode.In this mode, recognition won't start after <br>
you capture an image.By clicking the 'Batch Ocr',all the captured images will be recognized one time.<br>
6、Double click 'Page Ocr' to enable page-ocr mode.In this mode,after capturing a large image, <br>
all formulas in it will be box selection and you can copy a formula by clicking its box.<br>
You can see this video for some usage details: https://www.bilibili.com/video/BV137SdYCEd6 <br>

# How to use the code?
1.Setup your python env according to the requirements.txt <br>
2.Create a folder named "models", download and unzip models_texify.zip to the folder "models" from release page v1.7<br>
3.Create a folder named "texmml", download and unzip texmml.zip to the folder "texmml" from release page v1.2<br>
4.Just run the ocr_gui.py <br>
I've tried my best to simplify the code and preject,so it's easy to modify the GUI and intergrate other OCR model to this. <br>

# Potential issues
1、It will consume much time if you snip a very large image.<br>
2、Some format errors in MathML because it's tranformed from latex.I've tried to solve most of them.I suggest copying latex to mathtype directly<br>
3、Sometimes, there can be formatting issues with curly braces.<br>
4、Some letters or symbols cannot be recognized because not or rarely in the token vocabulary or training set.<br>
5、If you paste a very long formula to mathtype, it will report a error.So please avoid that.<br><br>
If you find any other issues，contact me.I'll keep updating this if necessary.

# Fequently Asked Questions
1、Support for Mac<br>
I'm trying to find a Mac to develop and test first but there is even no a Mac I can borrow.So I'll try a cloud Mac then.<br>
2、Self-defined Shortcut for Snipping<br>
I have set Alt+Q in the v1.4 for snipping which will satify most people.And probably I wont't add a self-define function,
because it'll introduce a config file which will make FMatPix.exe -> FMatPix.exe + xxx.xml. I mean there'll be an additional file.<br>
Forgive my OCD please.<br>
3、Support for double monitors<br>
From v1.5,FMatPix has started supporting double monitors.<br>

# A cup of coffee for the author
It is not very easy to develop the project, so if it is helpful to you, please consider offering me a cup of coffee🥤.<br>
<img src='https://github.com/user-attachments/assets/7ce31ebd-01fe-430b-8d73-d6be98e89d49' width = '150px' height='150px'>
