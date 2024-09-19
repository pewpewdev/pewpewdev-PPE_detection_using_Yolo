import xml.etree.ElementTree as ET
import os
import argparse

# Define the classes
classes = ["person"]

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(f"{output_dir}/{root.find('filename').text.split('.')[0]}.txt", 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Convert XML annotations to YOLO format.')
    parser.add_argument('xml_dir', type=str, help='Directory containing XML annotation files')
    parser.add_argument('output_dir', type=str, help='Directory to save YOLO format annotation files')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for xml_file in os.listdir(args.xml_dir):
        if xml_file.endswith(".xml"):
            convert_annotation(os.path.join(args.xml_dir, xml_file), args.output_dir)

if __name__ == '__main__':
    main()
