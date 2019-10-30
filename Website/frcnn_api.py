from flask import Flask, jsonify, make_response, request, render_template, flash,redirect
from flask_httpauth import HTTPBasicAuth
import test_frcnn as ult
import os
import numpy as np
from werkzeug.utils import secure_filename
import cv2
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
from keras_frcnn import roi_helpers
import test_mrcnn as mask_utl
import visualize
import skimage.io
import anaylse_result as AR

class mask_rcnn_Model:
    def __init__(self,model_name):
        self.model = mask_utl.loadmodel(model_name)
        self.class_names = mask_utl.defineClass(model_name)
        self.graph = tf.get_default_graph()
    def predict(self, filePath,filename,submodelID):
        with self.graph.as_default():
            image = skimage.io.imread(filePath)
            # Run detection
            results = self.model.detect([image], verbose=1)
            # Visualize results
            r = results[0]
            returnRes = visualize.display_instances(filename, submodelID,image=image, boxes=r['rois'],
                                                                          masks=r['masks'], class_ids=r['class_ids'],
                                                                          class_names=self.class_names, scores=r['scores'],
                                                                          show_mask=False)
            if returnRes != str(False):
                newPic_name, predict_classes, predict_scores = returnRes
                return newPic_name, predict_classes, predict_scores
            else:
                return 'Can not detect'

class frcnn_Model:
    def __init__(self, model_name):
        self.config = ult.define_C(model_name)
        self.model, self.modelClassify = self.loadmodel()
        self.graph = tf.get_default_graph()

    def loadmodel(self):
        if self.config.network == 'resnet50':
            import keras_frcnn.resnet as nn
        elif self.config.network == 'vgg':
            import keras_frcnn.vgg as nn
        if self.config.network == 'resnet50':
            num_features = 1024
        elif self.config.network == 'vgg':
            num_features = 512
        self.class_mapping = self.config.class_mapping

        if 'bg' not in self.class_mapping:
            self.class_mapping['bg'] = len(self.class_mapping)

        self.class_mapping = {v: k for k, v in self.class_mapping.items()}
        print(self.class_mapping)
        class_to_color = {self.class_mapping[v]: np.random.randint(0, 255, 3) for v in self.class_mapping}
        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.config.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(self.config.anchor_box_scales) * len(self.config.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, self.config.num_rois, nb_classes=len(self.class_mapping),
                                   trainable=True)

        model_rpn = Model(img_input, rpn_layers)
        model_classifier_only = Model([feature_map_input, roi_input], classifier)

        model_classifier = Model([feature_map_input, roi_input], classifier)

        print('Loading weights from {}'.format(self.config.model_path))
        model_rpn.load_weights(self.config.model_path, by_name=True)
        model_classifier.load_weights(self.config.model_path, by_name=True)

        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        return model_rpn, model_classifier_only

    def predict(self, img,filename):
        with self.graph.as_default():
            print(self.class_mapping)
            bbox_threshold = 0.8
            X, ratio = ult.format_img(img, self.config)

            if K.image_dim_ordering() == 'tf':
                X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            # print(X)
            [Y1, Y2, F] = self.model.predict(X)

            R = roi_helpers.rpn_to_roi(Y1, Y2, self.config, K.image_dim_ordering(), overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0] // self.config.num_rois + 1):
                ROIs = np.expand_dims(R[self.config.num_rois * jk:self.config.num_rois * (jk + 1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0] // self.config.num_rois:
                    # pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = self.modelClassify.predict([F, ROIs])

                for ii in range(P_cls.shape[1]):

                    if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                        tx /= self.config.classifier_regr_std[0]
                        ty /= self.config.classifier_regr_std[1]
                        tw /= self.config.classifier_regr_std[2]
                        th /= self.config.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append(
                        [self.config.rpn_stride * x, self.config.rpn_stride * y, self.config.rpn_stride * (x + w), self.config.rpn_stride * (y + h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []
            detect_imgs = []

            for key in bboxes:
                bbox = np.array(bboxes[key])
                count = 0
                newPic_name = "box_{}.jpg".format(str(filename[:-4] + str(count)))
                count += 1
                detect_imgs.append(newPic_name)
                original_img = cv2.imread('./static/tmp_pic/' + filename)
                height, width, _ = original_img.shape
                (resized_width, resized_height) = ult.get_new_img_size(width, height, 300)
                resize_img = cv2.resize(original_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("./static/img/doc/" + filename, resize_img)
                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                            overlap_thresh=0.5)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk, :]
                    (real_x1, real_y1, real_x2, real_y2) = ult.get_real_coordinates(ratio, x1, y1, x2, y2)
                    gt_x1, gt_x2 = real_x1 * (resized_width / width), real_x2 * (resized_width / width)
                    gt_y1, gt_y2 = real_y1 * (resized_height / height), real_y2 * (resized_height / height)
                    gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)
                    color = (0, 255, 0)
                    result_img = cv2.rectangle(resize_img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)

                    cv2.imwrite("./static/img/doc/" + newPic_name, result_img)
                    textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                    all_dets.append((key, 100 * new_probs[jk]))

                    (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    textOrg = (real_x1, real_y1 - 0)

                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            print(all_dets)
            if len(detect_imgs) > 0:
                print(detect_imgs[0])
                return detect_imgs[0], all_dets
            else:
                return "Can not detect"

global model_G,model_A,model_C,model_H,model_M,model_O

# model_G = frcnn_Model('G')
model_A = mask_rcnn_Model('A')
model_C = mask_rcnn_Model('C')
model_H = mask_rcnn_Model('H')
model_M = mask_rcnn_Model('M')
model_O = mask_rcnn_Model('O')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = os.urandom(24)
auth = HTTPBasicAuth()

UPLOAD_FOLDER = 'static/tmp_pic/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
else:
    pass
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/demo', methods=['POST','GET'])
def demo():
    return render_template('demo.html')

@app.route('/predict', methods=['POST'])
def face_insert():
    if 'imagefile' not in request.files:
        flash('No file part')
        return redirect('demo')
    upload_files = request.files['imagefile']
    if upload_files.filename == '':
        flash('No file selected for uploading')
        return redirect('demo')
    if upload_files and allowed_file(upload_files.filename):
        file = upload_files
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(image_path)
        submodelID = request.form.get("radio")
        print(submodelID)
        if submodelID == "A":
            result = model_A.predict(image_path,filename,submodelID)
        elif submodelID == "C":
            result = model_C.predict(image_path,filename,submodelID)
        elif submodelID == "H":
            result = model_H.predict(image_path,filename,submodelID)
        elif submodelID == "M":
            result = model_M.predict(image_path,filename,submodelID)
        elif submodelID == "O":
            result = model_O.predict(image_path,filename,submodelID)
        # else:
        #     result = model_G.predict(img, filename)
        if submodelID == "N" or submodelID == None:
            result_A = model_A.predict(image_path, filename,submodelID)
            returnList_A = AR.analyse(result_A, "A")
            result_C = model_C.predict(image_path, filename,submodelID)
            returnList_C = AR.analyse(result_C, "C")
            result_H = model_H.predict(image_path, filename,submodelID)
            returnList_H = AR.analyse(result_H, "H")
            result_M = model_M.predict(image_path, filename,submodelID)
            returnList_M = AR.analyse(result_M, "M")
            result_O = model_O.predict(image_path, filename,submodelID)
            returnList_O = AR.analyse(result_O, "O")
            returnList = AR.compareAllModel(returnList_A,returnList_C,returnList_H,returnList_M,returnList_O)
            if len(returnList) == 6:
                page, detectedImg, labels, scores, areas, other_cats = returnList
                return render_template("demo_all.html", original_img=filename, user_image=detectedImg,
                                       showLabels=labels, showScores=scores, areas=areas, other=other_cats, len=6)
            elif len(returnList) == 5:
                page, detectedImg, labels, scores, areas = returnList
                return render_template("demo_all.html", original_img=filename, user_image=detectedImg,
                                       showLabels=labels, showScores=scores, areas=areas, len=5)
            elif len(returnList) == 1:
                page, detectedImg, labels, scores, areas = returnList
                return render_template("demo_faild.html", user_image = 'notFound.png', showLabel ='Sorry, cannot detect this picture, maybe try again with another one')

        returnList = AR.analyse(result, submodelID)
        if returnList[0] == "demo_other.html":
            page, detectedImg, labels, scores, other_cats, area, floder = returnList
            return render_template("demo_other.html", original_img=filename, user_image=detectedImg,
                                   showLabels=labels, showScores=scores, other=other_cats, area=area, f=floder)
        elif returnList[0] == "demo_ok.html":
            page, detectedImg, labels, scores, area = returnList
            return render_template('demo_ok.html', original_img=filename, user_image=detectedImg, showLabels=labels,
                                   showScores=scores, area=area)
        elif returnList[0] == "demo_faild.html":
            return render_template("demo_faild.html", user_image = 'notFound.png', showLabel ='Sorry, cannot detect this picture, maybe try again with another one')

    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect('demo')


@auth.get_password
def get_password(username):
    if username == 'root':
        return 'root'
    return None


@auth.error_handler
def unauthorized():
    return render_template("demo_faild.html", user_image='notFound.png', showLabel="Unauthorized access")

@app.errorhandler(400)
def not_found(error):
    return render_template("demo_faild.html", user_image='notFound.png', showLabel="Invalid data!")

@app.errorhandler(500)
def internal_error(error):
    print(error)
    return render_template("demo_faild.html", user_image = 'notFound.png', showLabel ='Sorry, cannot detect this picture, maybe try again with another one')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

