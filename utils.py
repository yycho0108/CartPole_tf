import os
import signal
import sys
import argparse

def get_dir(*args):
    d = os.path.join(*args)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def directory_setup(task, run_id=None, **params):
    root = get_dir('data', task)
    run_root = get_dir(root, 'run')
    index = len(os.walk(run_root).next()[1])

    if run_id is None:
        run_id = str(index)
    else:
        run_id = '{}_{}'.format(index, run_id) 

    output_root = get_dir(run_root, run_id)
    output_graph = os.path.join(output_root, 'output_graph.pb')
    output_ckpt = os.path.join(output_root, 'model.ckpt')
    output_mem = os.path.join(output_root, 'memory.npy')
    log_root = get_dir('/tmp', task+'_logs')
    run_log_root = os.path.join(log_root, run_id)

    with open(os.path.join(output_root, 'run_id.txt'), 'wb') as f:
        f.write(run_id + '\n')
    with open(os.path.join(output_root, 'params.txt'), 'wb') as f:
        for (k,v) in params.iteritems():
            f.write('%s : %s\n' % (str(k), str(v)))
    return {
            'root' : root,
            'run_root' : run_root,
            'run_id' : run_id,
            'output_root' : output_root,
            'output_graph' : output_graph,
            'output_ckpt' : output_ckpt,
            'output_mem' : output_mem,
            'log_root' : log_root,
            'run_log_root' : run_log_root
            }

class StopRequest(object):
    def __init__(self):
        self._start = False
        self._stop = False
        signal.signal(signal.SIGINT, self.sig_cb)
    def start(self):
        self._start = True
    def stop(self):
        self._start = False
    def sig_cb(self, signal, frame):
        self._stop = True
        if not self._start:
            sys.exit(0)

