import cv2
import numpy as np

def class_name(classid):
    id_dict = {1:'Scratch', 2:'Dent', 3:'Shatter', 4:'Dislocation'}
    return id_dict[classid]

def damage_cost(classid):
    # cost_dict = {1: [800, 1400], 2:[1200, 3000],3:19000, 4:17000}
    cost_dict = {1: 900, 2:1600, 3:19000, 4:17000}

    return cost_dict[classid]

def area_ratio(image, roi, mask):
    y1, x1, y2, x2 =  tuple(roi)
    crop_mask = mask[y1:y1+(y2-y1),x1:x1+(x2-x1)].copy()
    pixels = cv2.countNonZero(np.float32(crop_mask))
    image_area = image.shape[0] * image.shape[1]
    area_ratio = 1 + (pixels / image_area)
    return area_ratio

def costEstimate(image, rois, masks, classids):
    cost_id_dict = {
    "Shatter": {"Count": 0, "Cost": 0},
    "Scratch": {"Count": 0, "Cost": 0},
    "Dent": {"Count": 0, "Cost": 0},
    "Dislocation": {"Count": 0, "Cost": 0}
    }
    total = 0
    count = int()
    cost_init = int()
    
    for index in range(rois.shape[0]):

        name = class_name(classids[index])
        cost = damage_cost(classids[index])
        ratio = area_ratio(image, rois[index], masks[: ,: ,index])

        total = total + round(cost * ratio,2)

        # unique_id = str()
        
        # for roi in rois[index]:
        #     unique_id = unique_id + str(roi)
            
        
        if name is 'Scratch':
            count = cost_id_dict[name]['Count'] + 1
            cost_init = cost_id_dict[name]['Cost'] + round(cost * ratio,2)
            cost_id_dict[name]['Count'] = count
            cost_id_dict[name]['Cost'] = cost_init
            # cost_id_dict[name] = "Range: Rs." + str(round(cost[0] * ratio,3)) + ' - Rs.' + str(round(cost[1] * ratio, 3))
        elif name is 'Dent':
            count = cost_id_dict[name]['Count'] + 1
            cost_init = cost_id_dict[name]['Cost'] + round(cost * ratio,2)
            cost_id_dict[name]['Count'] = count
            cost_id_dict[name]['Cost'] = cost_init
            # cost_id_dict[name] = "Range: Rs." + str(cost[0] * ratio) + ' - Rs.' + str(cost[1] * ratio)
        elif name is 'Shatter':
            count = cost_id_dict[name]['Count'] + 1
            cost_init = cost_id_dict[name]['Cost'] + round(cost * ratio,2)
            cost_id_dict[name]['Count'] = count
            cost_id_dict[name]['Cost'] = cost_init
            # cost_id_dict[name] = "Cost: Rs." + str(cost)
        else:
            count = cost_id_dict[name]['Count'] + 1
            cost_init = cost_id_dict[name]['Cost'] + round(cost * ratio,2)
            cost_id_dict[name]['Count'] = count
            cost_id_dict[name]['Cost'] = cost_init
            # cost_id_dict[name] = "Cost: Rs." + str(cost)

    for name, values in cost_id_dict.copy().items():
        if values['Count'] == 0:
            cost_id_dict.pop(name)

    return total, cost_id_dict



-----------------------------------------------------------------------------------------


# Mask-RCNN Imports
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# Matplotlib Import
import matplotlib.pyplot as plt

class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "damage"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 4
    # number of training steps per epoch
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = './model/mask_rcnn_damage_0011-2.h5'
model.load_weights(model_path, by_name=True)
print('Model Loaded Successfully!!')
model.keras_model._make_predict_function()

def model_predict(img_path):
	# image = load_img(img_path)
	# image = img_to_array(image)
	image = plt.imread(img_path)
	results = model.detect([image], verbose=1)
	return results


---------------------------------------------------------------------------------------


# Basic System Imports
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Flask Import
from flask import render_template, request, Flask, url_for


# Model Import
from app.utils import model_predict

# Cost Asssessment Import
from app import cost_assessment

# Mask-RCNN Import
from mrcnn import visualize

UPLOAD_PATH = 'static/uploads/'
UPLOAD_PRED_PATH = 'static/prediction/'

def base():
	return render_template('base.html')


def index():
	return render_template('index.html')


def damageapp():
	return render_template('damageapp.html')

def getwidth(path):
	img = Image.open(path)
	size = img.size # width and height
	aspect = size[0]/size[1] # width / height
	w = 300 * aspect
	return int(w)

def damage():
	fileupload = False
	cost_for_damage = True
	if request.method == 'POST':
		# File Upload
		fileupload=True
		f = request.files['fileToUpload']
		if not os.path.exists(os.path.join(UPLOAD_PATH, f.filename.split('.')[0])):
			os.mkdir(os.path.join(UPLOAD_PATH, f.filename.split('.')[0]))

		image_path = f.filename.split('.')[0] + '/' + f.filename

		# print(UPLOAD_PATH + image_path)

		f.save(UPLOAD_PATH + image_path)

		# print(f'File saved Successfully @ {image_path}')

		# Class Prediction
		results = model_predict(UPLOAD_PATH + image_path)

		class_names = ['BG', 'Scratch', 'Dent', 'Shatter', 'Dislocation']

		r = results[0]

		image = plt.imread(UPLOAD_PATH + image_path)
		# image = load_img(UPLOAD_PATH + image_path)
		# image = img_to_array(image)

		if not os.path.exists(UPLOAD_PRED_PATH + f.filename.split('.')[0]):
			os.mkdir(os.path.join(UPLOAD_PRED_PATH, f.filename.split('.')[0]))

		pred_path = UPLOAD_PRED_PATH + f.filename.split('.')[0]

		# Save Predicted Class Image
		visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,  r['scores'], path=pred_path + '/' + f.filename)
		get_masks_filenames = visualize.get_masks(image, r['masks'], r['rois'], class_names, r['class_ids'], path=pred_path + '/')
		top_masks_filenames = visualize.display_top_masks_edit(image, r['masks'], r['class_ids'], class_names, path=pred_path + '/')
		get_roi_filenames = visualize.get_rois(image, r['rois'], path=pred_path + '/')

		if cost_for_damage:
			total, cost = cost_assessment.costEstimate(image, r['rois'], r['masks'], r['class_ids'])
			print(f'File Successfully Manipulated @ {pred_path}')
			data = {
			'visualize': f.filename.split('.')[0] + '/' + f.filename,
			'width': getwidth(UPLOAD_PATH + f.filename.split('.')[0] + '/' + f.filename),
			'masks': get_masks_filenames,
			'top_masks': top_masks_filenames,
			'roi': get_roi_filenames,
			'cost': cost,
			'total': total,
			'tax': round(total * 0.1, 3),
			'tax_total': total + round(total * 0.1, 3)
			}
			return	render_template('damage.html', pagename='Damage Detect', fileupload=fileupload, data=data, cost_for_damage=cost_for_damage)
		else:
			data = {
			'visualize': f.filename.split('.')[0] + '/' + f.filename,
			'width': getwidth(UPLOAD_PATH + f.filename.split('.')[0] + '/' + f.filename),
			'masks': get_masks_filenames,
			'top_masks': top_masks_filenames,
			'roi': get_roi_filenames,
			}
			return	render_template('damage.html', pagename='Damage Detect', fileupload=fileupload, data=data, cost_for_damage=cost_for_damage)




		return render_template('damage.html', pagename='Damage Detect', fileupload=fileupload, data=data, cost_for_damage=cost_for_damage)
	return render_template('damage.html', pagename='Damage Detect', fileupload=fileupload)