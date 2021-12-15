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
# @return - metrics (2,) ndarray
def compute_metrics(contacts, CoM):
    regular = am.is_regular(contacts)
    distant = am.is_distant(contacts, CoM)
    #area =    am.is_area(contacts)

    metrics = np.array([regular, distant])
    return metrics

## This function reads as input a single csv and outputs the metrics for that csv
# @param filename - the name of the file to be outputted
#
# @return - metrics, jpg_filename
def read_single_csv(filename):
    #print(filename)
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
# @return metrics (n x k ndarray), filename (list n), jpg_files (list n)
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
    
    #indexes file name str to isolate grasp number (assumes ...##_2.csv naming convention)
    template_key['better'] = data[templates['path_ident']][path_num_b][-8:-6]
    
    template_key['worse'] = data[templates['path_ident']][path_num_w][-8:-6]
   
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


def exclude_function(shap, query_shap, data):
    q_succ = np.sum(query_shap) > data['thresh']
    o_succ = np.sum(shap) > data['thresh']

    return q_succ == o_succ



def main():
    parser = argparse.ArgumentParser(description='explainable grasp metrics')
    parser.add_argument('-t', type=str, default='train_files/', help='the folder of training csv and images')
    parser.add_argument('-q', type=str, default='test_files/35_2.csv', help='the query csv of shape points')
    parser.add_argument('--template', type=str, default='template_grasp.yaml', help='the yaml filename to load templates for generating the text explanation.')
    parser.add_argument('--num_alts', type=int, default=1, help='num alternatives (max number of features)')
    parser.add_argument('--thresh', type=float, default=1.45699, help='threshold to determine grasp')
    args = parser.parse_args()

    metrics, filenames, jpg_files = read_shape_points(args.t)
    query_metric, query_jpg = read_single_csv(args.q)

    print('query_metric')
    np.set_printoptions(suppress=True)
    print(query_metric)

    grasp_prediction = np.sum(query_metric) >= args.thresh
    if grasp_prediction:
        print('Query grasp is predicted to be successful')
    else:
        print('Query grasp is predicted to be unsuccessful')

    concat_metrics = np.append(metrics, query_metric[np.newaxis,:], axis=0)

    #### Perform the explainability selection
    alt_idxs, feat_idxs = gr.select_alts_from_shap_diff(\
                                concat_metrics.shape[0]-1, \
                                concat_metrics, \
                                args.num_alts+1, \
                                'similar_except', \
                                isMax=grasp_prediction,\
                                exclude_func=exclude_function,\
                                data={'thresh': args.thresh}) # ['random_less', 'worse_shap', 'similar_except']



    ###################### Explanation template generation
    explan_data = {}
    
    # Full labels are:'regularity of the grasp polygon', "distance between the center of object's mass and centroid of the grasp polygon", 'Area of the grasp polygon'
    explan_data['feature_labels'] = ['regularity', 'distance', 'area']
    explan_data['feature_idx'] = np.arange(query_metric.shape[0], dtype= int)
    explan_data['feature_types'] = ['metric']*2
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
    
    #declares fig to plot both images in a single window
    fig = plt.figure(figsize = (5, 7))
    
    #declares rows and columns for subplot function
    rows = 2
    columns = 1

    if len(query_jpg) > 0:
        img = mpimg.imread(query_jpg)
        
        fig.add_subplot(rows, columns, 1)
        
        plt.imshow(img)
        plt.axis('off')
        
        #assumes naming convention ...##_2.jpeg
        grasp_number = str(query_jpg[-9:-7])
        plt.title('Query Grasp: Grasp #' + grasp_number)
        
        show_at_least_one_image = True

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

        alt_predicted_to_be_successful = np.sum(concat_metrics[alt_idx]) >= args.thresh
        if alt_predicted_to_be_successful:
            print('\nThe alternative is predicted to be successful')
        else:
            print('\nThe alternative is predicted to be unsuccessful')
        explanations.append(explanation)

        # try to read and show alternative image
        if len(jpg_files[alt_idx]) > 0:
           
            img = mpimg.imread(jpg_files[alt_idx])
            
            #adds subplot in corresponding position
            fig.add_subplot(rows+i, columns, 2+i)
            
            plt.imshow(img)
            plt.axis('off')
                
            #assumes naming convention ...##_2.jpeg   
            alt_grasp_number = str(jpg_files[alt_idx][-9:-7])
            plt.title('Alternative Grasp Option '+str(i+1)+': Grasp #' + alt_grasp_number)
            
            
            show_at_least_one_image = True

    if show_at_least_one_image:
        plt.show()


if __name__ == '__main__':
    main()
