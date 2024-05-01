#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
from collections import defaultdict

from PIL import Image


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
                f.write(f'<th>start <input type="checkbox" onchange="hideColumn({z+2});"></th>')
                f.write(f'<th>end <input type="checkbox" onchange="hideColumn({z+3});"></th>')
                f.write(f'</tr></thead><tbody>')
                init = True

            meta_idx = int(data['meta_idx'])
            f.write(f'<tr>')
            f.write(f'<td>{str(meta_idx)}</td>')
            f.write(f'<td>{data["prompt"]}</td>')
            f.write(f'<td>{data["sequence"]}</td>')
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

