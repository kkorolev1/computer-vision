#!/usr/bin/env python3

from json import dumps, load
from os import environ
from os.path import join
from sys import argv


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res


def save_csv(img_classes, filename):
    with open(filename, 'w') as fhandle:
        print('filename,class_id', file=fhandle)
        for filename in sorted(img_classes.keys()):
            print('%s,%d' % (filename, img_classes[filename]), file=fhandle)


def check_test(data_dir):
    gt_dir = join(data_dir, 'gt')
    output_dir = join(data_dir, 'output')
    output = read_csv(join(output_dir, 'output.csv'))
    gt = read_csv(join(gt_dir, 'gt.csv'))

    correct = 0
    total = len(gt)
    for k, v in gt.items():
        if output[k] == v:
            correct += 1

    accuracy = correct / total

    res = 'Ok, accuracy %.4f' % accuracy
    if environ.get('CHECKER'):
        print(res)
    return res


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    result = results[-1]['status']

    if not result.startswith('Ok'):
        res = {'description': '', 'mark': 0}
    else:
        accuracy_str = result[13:]
        accuracy = float(accuracy_str)

        if accuracy >= 0.85:
            mark = 10
        elif accuracy >= 0.83:
            mark = 9
        elif accuracy >= 0.80:
            mark = 8
        elif accuracy >= 0.75:
            mark = 7
        elif accuracy >= 0.70:
            mark = 6
        elif accuracy >= 0.65:
            mark = 5
        elif accuracy >= 0.60:
            mark = 4
        elif accuracy >= 0.50:
            mark = 3
        elif accuracy >= 0.40:
            mark = 2
        elif accuracy > 0:
            mark = 1
        else:
            mark = 0

        res = {'description': accuracy_str, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


def run_single_test(data_dir, output_dir):
    from classification import train_classifier, classify
    from os.path import abspath, dirname, join

    train_dir = join(data_dir, 'train')
    test_dir = join(data_dir, 'test')

    train_gt = read_csv(join(train_dir, 'gt.csv'))
    train_img_dir = join(train_dir, 'images')

    model = train_classifier(train_gt, train_img_dir, fast_train=True)

    code_dir = dirname(abspath(__file__))
    model_path = join(code_dir, 'birds_model.ckpt')
    test_img_dir = join(test_dir, 'images')
    img_classes = classify(model_path, test_img_dir)
    save_csv(img_classes, join(output_dir, 'output.csv'))


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print('Usage: %s mode data_dir output_dir' % argv[0])
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally, run on dir with tests
        if len(argv) != 2:
            print('Usage: %s tests_dir' % argv[0])
            exit(0)

        from glob import glob
        from json import dump
        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs
        from os.path import basename, exists
        from shutil import copytree

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_*_input'))):
            output_dir = sub('input$', 'check', input_dir)
            run_output_dir = join(output_dir, 'output')
            makedirs(run_output_dir, exist_ok=True)
            gt_src = sub('input$', 'gt', input_dir)
            gt_dst = join(output_dir, 'gt')
            if not exists(gt_dst):
                copytree(gt_src, gt_dst)

            try:
                start = time()
                run_single_test(input_dir, run_output_dir)
                end = time()
                running_time = end - start
            except:
                status = 'Runtime error'
                traceback = format_exc()
            else:
                try:
                    status = check_test(output_dir)
                except:
                    status = 'Checker error'
                    traceback = format_exc()

            test_num = basename(input_dir)[:2]
            if status == 'Runtime error' or status == 'Checker error':
                print(test_num, status, '\n', traceback)
                results.append({'status': status})
            else:
                print(test_num, '%.2fs' % running_time, status)
                results.append({
                    'time': running_time,
                    'status': status})

        dump(results, open(join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print('Mark:', res['mark'], res['description'])
