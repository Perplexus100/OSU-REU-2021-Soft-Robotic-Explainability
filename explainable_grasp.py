# explainable_grasp.py
# Written Ian Rankin August 2021
#
# Takes as input grasp points and finds the best explanations for the grasp.

import argparse
import glob
import numpy as np
import oyaml as yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import analyze_metrics as am
import rdml_graph as gr
import pdb

## compute_metrics
# This function computes the given metrics and stores it as a numpy array
# @param contacts the input contact points
# @param CoM- the center of mass of the object
#
# @return - metrics (3,) ndarray
def compute_metrics(contacts, CoM):
    regular = am.is_regular(contacts)
    distant = am.is_distant(contacts, CoM)
    area =    am.is_area(contacts)

    metrics = np.array([regular, distant, area])
    return metrics

## This function reads as input a single csv and outputs the metrics for that csv
# @param filename - the name of the file to be outputted
#
# @return - metrics, jpg_filename
def read_single_csv(filename):
    jpg_filename = glob.glob(filename[:-4]+'.jp*g')
    if len(jpg_filename) > 0:
        jpg_filename = jpg_filename[0]
    else:
        jpg_filename = ''

    points = np.loadtxt(filename, delimiter=',')

    contacts = points[:3]
    CoM = points[3]
    CoG = points[4]

    metrics = compute_metrics(contacts, CoG)

    return metrics, jpg_filename



## read_shape_points
# this function reads all of the shape csv files in a directory and output the
# the metrics, filenames, and image filenames
# @param folder - the path to the folder
#
# @return metrics (n x 3 ndarray), filename (list n), jpg_files (list n)
def read_shape_points(folder):
    filenames = glob.glob(folder+'*.csv')

    metrics = None
    jpg_files = []

    #print(filenames)

    for i, filename in enumerate(filenames):
        metric, jpg_file = read_single_csv(filename)
        if i == 0:
            metrics = metric[np.newaxis, :]
        else:
            metrics = np.append(metrics, metric[np.newaxis,:], axis=0)
        jpg_files.append(jpg_file)

    #np.set_printoptions(suppress=True)
    #print(metrics)

    return metrics, filenames, jpg_files


# generates a text explanation when given a set of templates and choosen indicies
# of the selected feature.
# @param feat_idx - the index of the given feature in the local feature space
# @param idx_path_b - the index of the better path
# @param idx_path_w - the index of the worse path
# @param data the explan data needed to generate the explanations
# @param templates - the template data for generating the tex explanation.
# @param path_num_b - the path number of the better path used for refering to the
#                       paths (typically 0)
# @param path_num_w - the path number of the worse path.
def gen_explanation_with_given_decision( feat_idx, path_features, idx_path_b, idx_path_w, \
                                       data, templates, path_num_b, path_num_w):
    # find the type of feature for use with templates.
    feat_type = data['feature_types'][feat_idx]

    # feature_idx - the feature index in the global feature space.
    # feat_idx - is the feature index in the local features space provided to
    # to the decision tree. (may be filtered to only have features a path visited)
    feature_idx = data['feature_idx'][feat_idx]


    ################# Define the keys for each of the templates
    template_key = {}
    template_key['better'] = data[templates['path_ident']][path_num_b]
    template_key['worse'] = data[templates['path_ident']][path_num_w]
    template_key['feature'] = data['feature_labels'][feat_idx]

    # #
    # if feature_idx[0] >= len(templates['fields']) or feature_idx[0] < 0:
    #     template_key['field'] = templates['no_field']
    # else:
    #     template_key['field'] = templates['fields'][feature_idx[0]]

    header = templates['header']


    ################# Define the templates
    if feat_type in templates:
        # the feature type is in the set of templates
        feat_template = templates[feat_type]

        type_of_decision_node = feat_template['type']

        if type_of_decision_node == 'category':
            category = path_features[idx_path_b][feat_idx]
            template = feat_template['categories'][category]

        elif type_of_decision_node == 'float':
            if path_features[idx_path_b][feat_idx] > path_features[idx_path_w][feat_idx]:
                template = feat_template['larger']
            else:
                template = feat_template['smaller']

        ############# format templates using template key
        explanation = (header+template).format(**template_key)
        print(explanation)
        return feature_idx, explanation


    else:
        print('ERROR: was given a type of feature listed as: ' + str(feat_type) \
            + ' which is not in the given templates.')

        return None, None



def main():
    parser = argparse.ArgumentParser(description='explainable grasp metrics')
    parser.add_argument('-t', type=str, default='data/', help='the folder of training csv and images')
    parser.add_argument('-q', type=str, default='test/5_2.csv', help='the query csv of shape points')
    parser.add_argument('--template', type=str, default='template_grasp.yaml', help='the yaml filename to load templates for generating the text explanation.')
    parser.add_argument('--num_alts', type=int, default=1, help='num alternatives (max number of features)')
    args = parser.parse_args()

    metrics, filenames, jpg_files = read_shape_points(args.t)
    query_metric, query_jpg = read_single_csv(args.q)

    print('query_metric')
    np.set_printoptions(suppress=True)
    print(query_metric)

    concat_metrics = np.append(metrics, query_metric[np.newaxis,:], axis=0)

    #### Perform the explainability selection
    alt_idxs, feat_idxs = gr.select_alts_from_shap_diff(\
                                concat_metrics.shape[0]-1, \
                                concat_metrics, \
                                args.num_alts+1, \
                                'similar_except') # ['random_less', 'worse_shap', 'similar_except']



    ###################### Explanation template generation
    explan_data = {}
    explan_data['feature_labels'] = ['Regularity of the grasp polygon', "Daistance between the center of object's mass and centroid of the grasp polygon", 'Area of the grasp polygon'] 
    explan_data['feature_idx'] = np.arange(query_metric.shape[0], dtype=np.int)
    explan_data['feature_types'] = ['metric']*3
    explan_data['alt_names'] = filenames + [args.q]
    with open(args.template, 'rb') as f:
        templates_yaml = yaml.load(f.read(), Loader=yaml.Loader)


    print('alt_idxs')
    print(alt_idxs)
    print('feat_idxs')
    print(feat_idxs)


    #################### Visualization code

    explanations = []
    show_at_least_one_image = False

    for i, feat_idx in enumerate(feat_idxs):
        alt_idx = alt_idxs[i+1]
        _, explanation = gen_explanation_with_given_decision(\
                                    feat_idx, \
                                    concat_metrics, \
                                    alt_idxs[0], \
                                    alt_idx, \
                                    explan_data, \
                                    templates_yaml, \
                                    alt_idxs[0], \
                                    alt_idx)

        explanations.append(explanation)

        # try to read and show alternative image
        if len(jpg_files[alt_idx]) > 0:
            plt.figure()
            img = mpimg.imread(jpg_files[alt_idx])
            plt.imshow(img)
            show_at_least_one_image = True

    if show_at_least_one_image:
        plt.show()






if __name__ == '__main__':
    main()
