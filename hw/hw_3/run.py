#!/usr/bin/env python3

from math import floor
from os import environ
from os.path import join
from sys import argv


def check_test(data_dir):
    from pickle import load
    output_dir = join(data_dir, 'output')
    gt_dir = join(data_dir, 'gt')
    correct = 0
    with open(join(output_dir, 'output_seams'), 'rb') as fout, \
         open(join(gt_dir, 'seams'), 'rb') as fgt:
        for i in range(8):
            if load(fout) == load(fgt):
                correct += 1
    res = 'Ok %d/8' % correct
    if environ.get('CHECKER'):
        print(res)
    return res


def grade(data_path):
    from json import load, dumps
    results = load(open(join(data_path, 'results.json')))
    ok_count = 0
    for result in results:
        r = result['status']
        if r.startswith('Ok'):
            ok_count += int(r[3:4])
    total_count = len(results) * 8
    mark = floor(ok_count / total_count / 0.1)
    description = '%02d/%02d' % (ok_count, total_count)
    res = {'description': description, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


def run_single_test(data_dir, output_dir):
    from numpy import where
    from os.path import join
    from pickle import dump
    from seam_carve import seam_carve
    from skimage.io import imread

    def get_seam_coords(seam_mask):
        coords = where(seam_mask)
        t = [i for i in zip(coords[0], coords[1])]
        t.sort(key=lambda i: i[0])
        return tuple(t)

    def convert_img_to_mask(img):
        return ((img[:, :, 0] != 0) * -1 + (img[:, :, 1] != 0)).astype('int8')

    img = imread(join(data_dir, 'img.png'))
    mask = convert_img_to_mask(imread(join(data_dir, 'mask.png')))

    with open(join(output_dir, 'output_seams'), 'wb') as fhandle:
        for m in (None, mask):
            for direction in ('shrink', 'expand'):
                for orientation in ('horizontal', 'vertical'):
                    seam = seam_carve(img, orientation + ' ' + direction,
                                      mask=m)[2]
                    dump(get_seam_coords(seam), fhandle)


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
