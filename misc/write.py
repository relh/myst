#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
from collections import defaultdict

from PIL import Image

def br_insert(data):
    row_len = len(data)
    row = ''
    for i, x in enumerate(data):
        if row_len > 5 and i % (row_len // 5) == 0 and i != 0:
            row += '<br>'
        row += str(x)
    return row

def write_inference_html(all_data):
    with open(f'./outputs/index.html', 'w') as f:
        f.write('<html>')
        with open('web/header.html', 'r') as header:
            f.write(header.read())
        f.write('<body><table class="table" style="width: 100%; white-space: nowrap;"><thead><tr style="background-color:orange;">')

        init = False
        for data in all_data:
            if init is False:
                for z, x in enumerate(data.keys()):
                    f.write(f'<th>{x} <input type="checkbox" onchange="hideColumn({z+1});"></th>')
                f.write(f'<th>amount <input type="checkbox" onchange="hideColumn({z+2});"></th>')
                f.write(f'<th>start <input type="checkbox" onchange="hideColumn({z+3});"></th>')
                f.write(f'<th>end <input type="checkbox" onchange="hideColumn({z+4});"></th>')
                f.write(f'</tr></thead><tbody>')
                init = True

            meta_idx = int(data['meta_idx'])

            prompt = [x + ' ' for x in data["prompt"].split(' ')]
            sequence = [x for x in data["sequence"]]
            if isinstance(sequence[0], tuple):
                amount = [str(0.0) + ", " if x[1] is None else f"{x[1]:.2f}, " for x in sequence]
                sequence = [x[0] + ', ' for x in sequence]
            else:
                amount = [0.1]
                sequence = [x + ', ' for x in sequence]
            f.write(f'<tr>')
            f.write(f'<td><div>{str(meta_idx)}</div></td>')
            f.write(f'<td><div>{br_insert(prompt)}</div></td>')
            f.write(f'<td><div>{br_insert(sequence)}</div></td>')
            f.write(f'<td><div>{br_insert(amount)}</div></td>')
            f.write(f'<td><div><img class="lazy" onerror="this.onerror=null; this.remove();" data-src="/~relh/mysty/imgs/{meta_idx}_start.png"></div></td>')
            f.write(f'<td><div><img class="lazy" onerror="this.onerror=null; this.remove();" data-src="/~relh/mysty/imgs/{meta_idx}_end.png"></div></td>')
            f.write('</tr>')

        f.write('</tbody></table>')
        f.write('''<script>
                var lazyloadinstance = new LazyLoad({
                  // Your custom settings go here
                });
        </script>''')
        f.write('</div></body></html>')

if __name__ == "__main__":
    pickles = sorted([x for x in os.listdir('./outputs/pickles/') if 'pkl' in x], key=lambda x: -int(x.split('.')[0]))
    all_data = []
    for filename in pickles:
        all_data.append(pickle.load(open(f'./outputs/pickles/{filename}', 'rb')))

    write_inference_html(all_data)

