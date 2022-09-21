#!/usr/bin/env python3

from json import dumps, load
from os import environ
from os.path import join
from sys import argv, exit


def run_single_test(data_dir, output_dir):
    from align import align
    from numpy import ndarray
    from skimage.io import imread, imsave
    parts = open(join(data_dir, 'g_coord.csv')).read().rstrip('\n').split(',')
    g_coord = (int(parts[0]), int(parts[1]))
    img = imread(join(data_dir, 'img.png'), plugin='matplotlib')

    n_rows, n_cols = img.shape[:2]
    min_n_rows, min_n_cols = n_rows / 4.5, n_cols / 1.5
    aligned_img, (b_row, b_col), (r_row, r_col) = align(img, g_coord)

    assert type(aligned_img) is ndarray, 'aligned image is not ndarray'
    n_rows, n_cols = aligned_img.shape[:2]
    assert n_rows > min_n_rows and n_cols > min_n_cols, 'aligned image is too small'

    with open(join(output_dir, 'output.csv'), 'w') as fhandle:
        print('%d,%d,%d,%d' % (b_row, b_col, r_row, r_col), file=fhandle)

    imsave(join(output_dir, 'aligned_img.png'), aligned_img)


def check_test(data_dir):

    with open(join(data_dir, 'output/output.csv')) as fhandle:
        parts = fhandle.read().rstrip('\n').split(',')
        b_row, b_col, r_row, r_col = map(int, parts)

    with open(join(data_dir, 'gt/gt.csv')) as fhandle:
        parts = fhandle.read().rstrip('\n').split(',')
        coords = map(int, parts[1:])
        gt_b_row, gt_b_col, _, _, gt_r_row, gt_r_col, diff_max = coords

    diff = abs(b_row - gt_b_row) + abs(b_col - gt_b_col) + \
        abs(r_row - gt_r_row) + abs(r_col - gt_r_col)

    if diff > diff_max:
        res = 'Wrong answer'
    else:
        res = 'Ok'
    if environ.get('CHECKER'):
        print(res)
    return res


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    ok_count = 0
    for result in results:
        if result['status'] == 'Ok':
            ok_count += 1
    total_count = len(results)
    description = '%02d/%02d' % (ok_count, total_count)
    mark = ok_count / total_count * 10
    res = {'description': description, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print(f'Usage: {argv[0]} mode data_dir output_dir')
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
            print(f'Usage: {argv[0]} tests_dir')
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
                print(test_num, f'{running_time:.2f}s', status)
                results.append({
                    'time': running_time,
                    'status': status})

        dump(results, open(join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print('Mark:', res['mark'], res['description'])
