#!/usr/bin/env python3
"""Render source-anchored component SVGs in the selected central-hub format."""
from __future__ import annotations

import argparse
import subprocess
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

from render_overview_software_stack_components import Diagram, preview
from software_stack_component_hub_data import DIAGRAMS, PINS, REPOSITORIES, SOURCES

ROOT = Path(__file__).resolve().parents[1]


def source_metadata(root, specs):
    names = {s['core_ref'] for s in specs}
    for spec in specs:
        for group in ('definitions', 'producers', 'transforms', 'consumers'):
            for node in spec[group]:
                names.update(node['refs'])
    repos = {SOURCES[name][0] for name in names}
    locked = {parts[1]: parts[3] for line in (root/'upstream-sources.lock').read_text().splitlines()
              if line and not line.startswith('#') and len(parts := line.split('|')) >= 4}
    for repo in repos:
        head = subprocess.check_output(['git', '-C', str(root/'upstream'/repo), 'rev-parse', 'HEAD'], text=True).strip()
        if head != PINS[repo] or locked['upstream/'+repo] != PINS[repo]:
            raise ValueError(f'{repo}: source revision changed; review content and anchors')
    files, anchors = {}, {}
    for name in sorted(names):
        repo, path, needle, fixed = SOURCES[name]
        key = (repo, path)
        if key not in files:
            data = subprocess.check_output(['git', '-C', str(root/'upstream'/repo), 'show', f'{PINS[repo]}:{path}'])
            local = root/'upstream'/repo/path
            if local.exists() and local.read_bytes() != data:
                raise ValueError(f'Modified source: {repo}/{path}')
            files[key] = data.decode().splitlines()
        lines = files[key]
        matches = [i+1 for i, line in enumerate(lines) if needle in line]
        if fixed is not None:
            if fixed not in matches:
                raise ValueError(f'Moved anchor {name}: expected {fixed}, got {matches}')
            line = fixed
        elif len(matches) == 1:
            line = matches[0]
        else:
            raise ValueError(f'Ambiguous anchor {name}: {matches}')
        anchors[name] = dict(repo=repo, path=path, line=line, needle=needle,
                             href=f'https://github.com/{REPOSITORIES[repo]}/blob/{PINS[repo]}/{path}#L{line}')
    return dict(source_pins={r: PINS[r] for r in sorted(repos)}, source_anchors=anchors,
                evidence='Pinned source inspection and SVG validation only; no device execution',
                sparse_source_policy='Read absent working-tree files from the pinned Git objects; no checkout changes')


class HubDiagram(Diagram):
    def __init__(self, spec, metadata):
        super().__init__(10300, 6700, spec['title'], '', metadata)
        self.spec = spec
        self.base, self.edges, self.nodes, self.labels = [], [], [], []
        self.text_boxes, self.edge_boxes = [], []
        self.bg, self.card = '#fafbf8', '#ffffff'
        self.ink, self.muted = '#203344', '#52687b'
        self.colors.update(program='#2866a4', data='#087e78', transform='#8656ac', consume='#ae622c', neutral='#788b9d')
        self.roles, self.relations = {}, []
        self.text(80, 100, spec['title'], 50, bold=True)
        self.text(83, 160, spec['summary'], 27, self.muted)
        for x, label, color in [(90,'核心抽象','program'),(620,'产生','data'),(1060,'变换','transform'),
                                (1500,'消费','consume'),(1940,'灰虚线：定义 / 约束','neutral'),(2760,'棕虚线：待确认关系','unknown')]:
            self.labels.append(f'<path d="M{x},229h58" stroke="{self.colors[color]}" stroke-width="5"/>')
            self.text(x+76, 239, label, 25, self.colors[color])
        self.text(85, 300, '阅读方式：先看中央研究对象，再沿左侧来源、上方定义、下方变换与右侧消费者展开；分支不代表统一固定流水线。', 25, self.muted)
        self.base.append('<rect x="40" y="355" width="10220" height="5980" rx="30" fill="none" stroke="#9eafbd" stroke-width="4"/>')

    def group(self, key, x, y, w, h, title, subtitle, role):
        self.roles[key] = role
        self.base.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="#f0f4f7" stroke="#b1c3d2" stroke-width="2"/>')
        self.text(x+30, y+55, title, 35, bold=True)
        for i, line in enumerate(self.wrap(subtitle, w-60, 24)):
            self.text(x+32, y+101+i*34, line, 24, self.muted)

    def refs(self, key, x, y, w, h, refs):
        for i, name in enumerate(refs):
            a = self.meta['source_anchors'][name]
            label = f'↗ {name} · {Path(a["path"]).name}:{a["line"]}'
            if self.measure(label, 19)[1].width > w-50:
                label = f'↗ {name} · L{a["line"]}'
            self.text(x+24, y+h-23-29*(len(refs)-1-i), label, 19, self.colors['neutral'], href=a['href'], owner=key)

    def box(self, key, x, y, w, h, data, role):
        self.roles[key] = role
        color = {'definition':'neutral','producer':'data','transform':'transform','consumer':'consume','result':'neutral'}[role]
        if data.get('uncertain'):
            color = 'unknown'
        tags = {'definition':'定义 / 约束','producer':'产生：输入 → 核心抽象','transform':'变换：核心抽象 → 新对象或状态',
                'consumer':'消费：读取核心抽象','result':'消费结果'}
        self.node(key,x,y,w,h,data.get('tag') or tags[role],data['title'],data['rows'],color,dashed=data.get('uncertain',False))
        self.refs(key,x,y,w,h,data.get('refs',()))

    def core(self, key, x, y, w, h, title, rows):
        self.roles[key] = 'core'
        self.boxes[key] = x,y,w,h
        dash = ' stroke-dasharray="14 10"' if self.spec.get('uncertain') else ''
        self.nodes.append(f'<rect id="{key}" x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="#e3effd" stroke="#2866a4" stroke-width="5"{dash}/>')
        self.text(x+30,y+38,'研究对象 · '+('动作与状态' if self.spec.get('action') else '核心抽象'),21,self.colors['program'],bold=True,owner=key)
        self.text(x+30,y+111,title,46,bold=True,owner=key)
        baseline = y+176
        for row in rows:
            for line in self.wrap(row,w-65,25):
                self.text(x+30,baseline,line,25,self.muted,owner=key)
                baseline += 40
        self.refs(key,x,y,w,h,(self.spec['core_ref'],))

    def relation(self, source, target, kind, points, uncertain=False):
        pairs={'define':('definition','core'),'produce':('producer','core'),'transform_in':('core','transform'),
               'transform_out':('transform','core'),'consume':('core','consumer'),'result':('consumer','result'),
               'reuse':('core','core')}
        if kind in pairs and (self.roles[source], self.roles[target]) != pairs[kind]:
            raise ValueError(f'Invalid core abstraction relation: {source} -> {target}')
        uncertain = uncertain or self.spec.get('uncertain',False)
        color={'define':'neutral','produce':'data','transform_in':'transform','transform_out':'transform','consume':'consume','result':'neutral','reuse':'transform'}[kind]
        self.edge(points,'unknown' if uncertain else color,dashed=uncertain or kind=='define')
        self.relations.append(dict(source=source,target=target,kind=kind,evidence='unconfirmed boundary association' if uncertain else 'source-grounded'))

    def save(self, path):
        for key, role in self.roles.items():
            if role == 'transform':
                assert any(r['target']==key and r['kind']=='transform_in' for r in self.relations)
                assert any(r['source']==key and r['kind']=='transform_out' for r in self.relations)
            elif role == 'consumer':
                assert any(r['target']==key and r['kind']=='consume' for r in self.relations)
            elif role == 'producer':
                assert any(r['source']==key and r['kind']=='produce' for r in self.relations)
        self.meta.update(component=self.spec['key'],core_abstraction=self.spec['core'],node_roles=self.roles,
                         diagram_relations=self.relations,private_implementation_unconfirmed=self.spec.get('uncertain',False))
        super().save(path)
        xml = path.read_text()
        start = xml.index('<desc id="desc">')+len('<desc id="desc">')
        end = xml.index('</desc>',start)
        path.write_text(xml[:start]+escape(self.spec['summary']+' 四个视角均围绕核心抽象；灰虚线表示定义约束，棕虚线标记待确认关系。')+xml[end:])
        ET.parse(path)


def render(spec, metadata):
    d=HubDiagram(spec,metadata)
    d.group('definitions',3060,410,2960,1410,'定义 · 结构、语义与约束','上方说明中央对象是什么；定义连线不是执行顺序。','panel')
    for i,n in enumerate(spec['definitions']):
        x,y=3140+1430*(i%2),620+570*(i//2)
        d.box(f'd{i}',x,y,1350,440,n,'definition')
    d.core('core',3380,2200,2300,700,spec['core'],spec['center'])
    for i in range(len(spec['definitions'])):
        x,y=3140+1430*(i%2),620+570*(i//2)
        lane=1125 if i<2 else 1750
        d.relation(f'd{i}','core','define',[(x+675,y+440),(x+675,lane),(4530,lane),(4530,2200)])
    d.label(4600,2040,'共同定义中央研究对象','neutral',size=24)

    subtitle='输入与构造机制明确说明产物；这些入口可独立或组合使用。'
    if spec.get('uncertain'): subtitle='公开编译边界可见；构造 LLO 的内部函数和输入输出仍待确认。'
    d.group('producers',100,920,2600,2560,'产生 · 从哪里得到它',subtitle,'panel')
    for i,n in enumerate(spec['producers']):
        y=1180+760*i
        d.box(f'p{i}',220,y,2360,440,n,'producer')
        d.relation(f'p{i}','core','produce',[(2580,y+220),(2850,y+220),(2850,2470),(3380,2470)],n.get('uncertain',False))
    d.label(2890,2420,'产生 / 构造' if not spec.get('uncertain') else '内部路径待确认','data' if not spec.get('uncertain') else 'unknown',size=24)

    d.group('consumers',6270,770,3910,2750,'消费 · 谁读取它，得到什么','每条橙色路径从中央对象出发；右侧单列结果及语义边界。' if not spec.get('uncertain') else '虚线表示证据边界；公开返回值不能反推 LLO 消费者。','panel')
    for i,n in enumerate(spec['consumers']):
        y=1010+580*i
        d.box(f'c{i}',6390,y,1580,420,n,'consumer')
        d.box(f'r{i}',8490,y,1550,420,dict(title=n['result'],rows=n['result_rows'],uncertain=n.get('uncertain',False)),'result')
        d.relation('core',f'c{i}','consume',[(5680,2520),(6030,2520),(6030,y+190),(6390,y+190)],n.get('uncertain',False))
        d.relation(f'c{i}',f'r{i}','result',[(7970,y+190),(8490,y+190)],n.get('uncertain',False))
        d.label(8140,y+155,'得到' if not spec.get('uncertain') else '边界结果','neutral',size=24)

    sub='每个操作以中央对象为输入，返回同类对象、修改后的表示或新的生命周期状态；按需组合。'
    if spec.get('uncertain'): sub='四个待取证问题；没有声称已识别 LLO pass 或已获得相应变换产物。'
    d.group('transforms',2750,3930,4310,2350,'变换 · 如何改变这个抽象',sub,'panel')
    for i,n in enumerate(spec['transforms']):
        x,y=2880+2080*(i%2),4180+680*(i//2)
        key=f't{i}'
        d.box(key,x,y,1860,450,n,'transform')
        d.relation('core',key,'transform_in',[(4170,2900),(4170,3720),(2650,3720),(2650,4070),(x-45,4070),(x-45,y+170),(x,y+170)],n.get('uncertain',False))
    d.core('after',3740,5750,2360,410,spec['after'],spec['after_rows'])
    for i,n in enumerate(spec['transforms']):
        x,y=2880+2080*(i%2),4180+680*(i//2)
        d.relation(f't{i}','after','transform_out',[(x+1860,y+290),(x+1910,y+290),(x+1910,5530),(4920,5530),(4920,5750)],n.get('uncertain',False))
    if spec['key'] not in {'tpu','runtime','pjrt'}:
        d.relation('after','core','reuse',[(3740,5960),(2450,5960),(2450,3620),(3170,3620),(3170,2800),(3380,2800)])
        d.label(2490,3570,'同类对象可继续消费或再次变换','transform',size=24)
    for i,line in enumerate(d.wrap(spec['foot'],10030,25)):
        d.text(85,6400+36*i,line,25,d.muted)
    pins=' · '.join(f'{repo} {pin[:12]}' for repo,pin in metadata['source_pins'].items())
    d.text(85,6540,'固定源码：'+pins+'；图内链接定位完整提交、文件和行号。',23,d.muted)
    d.text(85,6580,'证据：固定源码检查 + SVG 布局校验；不代表 CPU / GPU / TPU 执行、数值正确性或性能验证。',23,d.muted)
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--preview-dir',type=Path)
    p.add_argument('--component',choices=[s['key'] for s in DIAGRAMS])
    args=p.parse_args()
    specs=[s for s in DIAGRAMS if not args.component or s['key']==args.component]
    all_meta=source_metadata(args.root,specs)
    output=args.output_dir or args.root/'research/software-stack'
    for spec in specs:
        names={spec['core_ref']}
        for group in ('definitions','producers','transforms','consumers'):
            for n in spec[group]: names.update(n['refs'])
        meta=dict(all_meta,source_anchors={k:all_meta['source_anchors'][k] for k in sorted(names)})
        used_repos={a['repo'] for a in meta['source_anchors'].values()}
        meta['source_pins']={r:v for r,v in all_meta['source_pins'].items() if r in used_repos}
        path=output/spec['path']; path.parent.mkdir(parents=True,exist_ok=True)
        render(spec,meta).save(path)
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True,exist_ok=True)
            preview(path,args.preview_dir/(spec['key']+'.png'),width=2400)
    print(f'{len(specs)} diagrams; {len(all_meta["source_anchors"])} verified source anchors')


if __name__=='__main__':
    main()
