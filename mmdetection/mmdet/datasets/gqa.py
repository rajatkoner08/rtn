# Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py # noqa
# and https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa

import glob
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.utils import print_log
from PIL import Image
import cv2
import json
import pickle
from tqdm import tqdm

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


#root_dir = os.path.dirname(os.path.abspath(__file__)).rsplit('/',3)[0]  # set root dir as root for scene graph
# changed by Deepan (3 changes to 2)
root_dir = os.path.dirname(os.path.abspath(__file__)).rsplit('/',3)[0]

@DATASETS.register_module()
class GQADataset(CustomDataset):
    CLASSES = pickle.load(open(os.path.join(root_dir, 'data/gqa/ind_to_class.pkl'), 'rb'))
    cat2label = {name: i for i, name in enumerate(CLASSES)}
    count = 0


    def load_annotations(self, ann_info):
        """load bbox  annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks.
        """
        with open(os.path.join(root_dir, ann_info), 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        self.filenames = list(content.keys())
        gt_attr = []  # todo add attribute
        data_infos = []
        # get all info from file
        for img_no, item in content.items():
            img_w = item['width']
            img_h = item['height']
            boxes = np.array(item['boxes']).astype(np.float32)
            lables = np.asarray(item['classes'])[:, 1]
                        
            gt_attr = item['attributes']
            # NumAttr = 200
            gt_attr_dist = np.zeros((len(lables), 200 + 1), dtype=np.int64)
            for i in range(len(lables)):
                # gt_attr_dist[i, 0] = i
                if i in gt_attr.keys():
                    # gt_attr_dist[i, :][np.asarray(gt_attr[i]) + 1] = 1
                    gt_attr_dist[i, :][np.asarray(gt_attr[i])] = 1
            
            data_infos.append(
                dict(
                    filename=img_no + '.jpg',
                    width=img_w,
                    height=img_h,
                    ann=dict(
                        bboxes=boxes,
                        # labels=lables,
                        # attr=gt_attr_dist
                        labels = np.concatenate((np.array([lables]).T,gt_attr_dist),axis=1)# if 'train' in ann_info else lables # rtodo create support for val
                    )))
        
        return data_infos

    def get_ann_info(self, idx):
        self.count +=1
        return self.data_infos[idx]['ann']



    def results2txt(self, results, outfile_prefix):
        """Dump the detection results to a txt file.

        Args:
            results (list[list | tuple]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files.
                If the prefix is "somepath/xxx",
                the txt files will be named "somepath/xxx.txt".

        Returns:
            list[str: str]: result txt files which contains corresponding
            instance segmentation images.
        """
        try:
            import cityscapesscripts.helpers.labels as CSLabels
        except ImportError:
            raise ImportError('Please run "pip install citscapesscripts" to '
                              'install cityscapesscripts first.')
        result_files = []
        os.makedirs(outfile_prefix, exist_ok=True)
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.data_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            pred_txt = osp.join(outfile_prefix, basename + '_pred.txt')

            bbox_result, segm_result = result
            bboxes = np.vstack(bbox_result)
            # segm results
            if isinstance(segm_result, tuple):
                # Some detectors use different scores for bbox and mask,
                # like Mask Scoring R-CNN. Score of segm will be used instead
                # of bbox score.
                segms = mmcv.concat_list(segm_result[0])
                mask_score = segm_result[1]
            else:
                # use bbox score for mask score
                segms = mmcv.concat_list(segm_result)
                mask_score = [bbox[-1] for bbox in bboxes]
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            assert len(bboxes) == len(segms) == len(labels)
            num_instances = len(bboxes)
            prog_bar.update()
            with open(pred_txt, 'w') as fout:
                for i in range(num_instances):
                    pred_class = labels[i]
                    classes = self.CLASSES[pred_class]
                    class_id = CSLabels.name2label[classes].id
                    score = mask_score[i]
                    mask = maskUtils.decode(segms[i]).astype(np.uint8)
                    png_filename = osp.join(outfile_prefix,
                                            basename + f'_{i}_{classes}.png')
                    mmcv.imwrite(mask, png_filename)
                    fout.write(f'{osp.basename(png_filename)} {class_id} '
                               f'{score}\n')
            result_files.append(pred_txt)

        return result_files

    def format_results(self, results, txtfile_prefix=None):
        """Format the results to txt (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving txt/png files when txtfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2txt(results, txtfile_prefix)

        return result_files, tmp_dir

    # def evaluate(self,
    #              results,
    #              metric='bbox',
    #              logger=None,
    #              outfile_prefix=None,
    #              classwise=False,
    #              proposal_nums=(100, 300, 1000),
    #              iou_thrs=np.arange(0.5, 0.96, 0.05)):
    #     """Evaluation in Cityscapes/COCO protocol.
    #
    #     Args:
    #         results (list[list | tuple]): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated. Options are
    #             'bbox', 'segm', 'proposal', 'proposal_fast'.
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #         outfile_prefix (str | None): The prefix of output file. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If results are evaluated with COCO protocol, it would be the
    #             prefix of output json file. For example, the metric is 'bbox'
    #             and 'segm', then json files would be "a/b/prefix.bbox.json" and
    #             "a/b/prefix.segm.json".
    #             If results are evaluated with cityscapes protocol, it would be
    #             the prefix of output txt/png files. The output files would be
    #             png images under folder "a/b/prefix/xxx/" and the file name of
    #             images would be written into a txt file
    #             "a/b/prefix/xxx_pred.txt", where "xxx" is the video name of
    #             cityscapes. If not specified, a temp file will be created.
    #             Default: None.
    #         classwise (bool): Whether to evaluating the AP for each class.
    #         proposal_nums (Sequence[int]): Proposal number used for evaluating
    #             recalls, such as recall@100, recall@1000.
    #             Default: (100, 300, 1000).
    #         iou_thrs (Sequence[float]): IoU threshold used for evaluating
    #             recalls. If set to a list, the average recall of all IoUs will
    #             also be computed. Default: 0.5.
    #
    #     Returns:
    #         dict[str, float]: COCO style evaluation metric or cityscapes mAP
    #             and AP@50.
    #     """
    #     eval_results = dict()
    #
    #     metrics = metric.copy() if isinstance(metric, list) else [metric]
    #
    #     if 'cityscapes' in metrics:
    #         eval_results.update(
    #             self._evaluate_cityscapes(results, outfile_prefix, logger))
    #         metrics.remove('cityscapes')
    #
    #     # left metrics are all coco metric
    #     if len(metrics) > 0:
    #         # create CocoDataset with CityscapesDataset annotation
    #         self_coco = CocoDataset(self.ann_file, self.pipeline.transforms,
    #                                 None, self.data_root, self.img_prefix,
    #                                 self.seg_prefix, self.proposal_file,
    #                                 self.test_mode, self.filter_empty_gt)
    #         # TODO: remove this in the future
    #         # reload annotations of correct class
    #         self_coco.CLASSES = self.CLASSES
    #         self_coco.data_infos = self_coco.load_annotations(self.ann_file)
    #         eval_results.update(
    #             self_coco.evaluate(results, metrics, logger, outfile_prefix,
    #                                classwise, proposal_nums, iou_thrs))
    #
    #     return eval_results

    # def _evaluate_cityscapes(self, results, txtfile_prefix, logger):
    #     """Evaluation in Cityscapes protocol.
    #
    #     Args:
    #         results (list): Testing results of the dataset.
    #         txtfile_prefix (str | None): The prefix of output txt file
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #
    #     Returns:
    #         dict[str: float]: Cityscapes evaluation results, contains 'mAP'
    #             and 'AP@50'.
    #     """
    #
    #     try:
    #         import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa
    #     except ImportError:
    #         raise ImportError('Please run "pip install citscapesscripts" to '
    #                           'install cityscapesscripts first.')
    #     msg = 'Evaluating in Cityscapes style'
    #     if logger is None:
    #         msg = '\n' + msg
    #     print_log(msg, logger=logger)
    #
    #     result_files, tmp_dir = self.format_results(results, txtfile_prefix)
    #
    #     if tmp_dir is None:
    #         result_dir = osp.join(txtfile_prefix, 'results')
    #     else:
    #         result_dir = osp.join(tmp_dir.name, 'results')
    #
    #     eval_results = {}
    #     print_log(f'Evaluating results under {result_dir} ...', logger=logger)
    #
    #     # set global states in cityscapes evaluation API
    #     CSEval.args.cityscapesPath = os.path.join(self.img_prefix, '../..')
    #     CSEval.args.predictionPath = os.path.abspath(result_dir)
    #     CSEval.args.predictionWalk = None
    #     CSEval.args.JSONOutput = False
    #     CSEval.args.colorized = False
    #     CSEval.args.gtInstancesFile = os.path.join(result_dir,
    #                                                'gtInstances.json')
    #     CSEval.args.groundTruthSearch = os.path.join(
    #         self.img_prefix.replace('leftImg8bit', 'gtFine'),
    #         '*/*_gtFine_instanceIds.png')
    #
    #     groundTruthImgList = glob.glob(CSEval.args.groundTruthSearch)
    #     assert len(groundTruthImgList), 'Cannot find ground truth images' \
    #         f' in {CSEval.args.groundTruthSearch}.'
    #     predictionImgList = []
    #     for gt in groundTruthImgList:
    #         predictionImgList.append(CSEval.getPrediction(gt, CSEval.args))
    #     CSEval_results = CSEval.evaluateImgLists(predictionImgList,
    #                                              groundTruthImgList,
    #                                              CSEval.args)['averages']
    #
    #     eval_results['mAP'] = CSEval_results['allAp']
    #     eval_results['AP@50'] = CSEval_results['allAp50%']
    #     if tmp_dir is not None:
    #         tmp_dir.cleanup()
    #     return eval_results

def get_im_size(ann_path=None):
    #     """Filter images too small or without ground truths."""
        graphs = json.load(open('/nfs/data3/koner/train_sceneGraphs.json'))

        heights, widths = [], []
        for k, v in graphs.items():
            heights.append(v['height'])
            widths.append(v['width'])

        print('max hight : ',np.max(heights), ' max width : ',np.max(widths))

if __name__ == "__main__":
    get_im_size()
    ds = GQADataset(ann_file='test', pipeline=None)
    ds.parse_ann_info('test')
