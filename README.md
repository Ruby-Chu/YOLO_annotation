# Convert WIDER Face annotations to YOLO format.

#YOLO #annotation 

將[WIDER Face](http://shuoyang1213.me/WIDERFACE/index.html)的標註資料轉換成yolo的格式。

**YOLO標註格式**
```bash
<class_id> <x_center> <y_center> <width> <height>
```

**WIDER Face 標註格式**
```bash
Attached the mappings between attribute names and label values.

blur:
  clear->0
  normal blur->1
  heavy blur->2

expression:
  typical expression->0
  exaggerate expression->1

illumination:
  normal illumination->0
  extreme illumination->1

occlusion:
  no occlusion->0
  partial occlusion->1
  heavy occlusion->2

pose:
  typical pose->0
  atypical pose->1

invalid:
  false->0(valid image)
  true->1(invalid image)

The format of txt ground truth.
File name
Number of bounding box
x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
```

1. 先下載WIDER_face的影像資料，並解壓縮。