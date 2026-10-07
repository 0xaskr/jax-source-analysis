#!/usr/bin/env python3
"""Render component workflows in the overview style, with SVG-only output."""
from __future__ import annotations

import argparse
from html import escape
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from render_overview_software_stack_components import Diagram
from render_overview_software_stack_flows import orthogonal_route, compact, overlaps
from component_diagram_sources import source_metadata
from software_stack_component_flow_data import diagrams, LABELS, PATHS, NEIGHBORS, OVERVIEW_IDS

ROOT = Path(__file__).resolve().parents[1]
LEGEND = [('编译 / 程序','program'), ('执行 / 输入数据','data'), ('程序变换','transform'),
          ('编译结果返回','result'), ('输出对象返回','output'), ('完成 / 就绪状态','completion'),
          ('结构 / 持有 / 观察','neutral'), ('内部证据边界','unknown')]
DASHED = {'result','completion','neutral','unknown'}
VIEW_IDS = ('view_definition','view_production','view_transformation','view_consumption')


def component_view_plan(spec):
    """Size generic component blocks from their row counts and column counts.

    This is intentionally separate from the fixed JAX layout: changing another
    component's content must not move JAX's existing nodes or flow captions.
    """
    parts={part['id']:part for part in spec['sections']}
    if len(parts)!=len(spec['sections']) or not set(VIEW_IDS)<=parts.keys():
        raise ValueError('component_views requires four uniquely named research views')
    if set(parts)-{*VIEW_IDS,'downstream'}:
        raise ValueError('Unexpected component view ID')
    plan={};x=220
    for ident in VIEW_IDS:
        rows=parts[ident]['rows'];cols=max(1,max((len(row) for row in rows),default=1))
        if cols>3:raise ValueError('Research view rows support at most three cards')
        card_width=2340 if ident=='view_definition' else 1720
        width=440+cols*card_width+max(0,cols-1)*440
        plan[ident]=dict(x=x,width=width,columns=max(1,cols),card_width=card_width)
        x+=width+(580 if ident=='view_definition' else 500)
    operation_start=plan['view_production']['x']
    operation_right=plan['view_consumption']['x']+plan['view_consumption']['width']
    right=operation_right
    if 'downstream' in parts:
        if not parts['downstream'].get('auxiliary'):
            raise ValueError('The downstream panel must be marked auxiliary')
        cols=max(1,max((len(row) for row in parts['downstream']['rows']),default=1))
        if cols>5:raise ValueError('Auxiliary rows support at most five cards')
        width=max(operation_right-operation_start,440+cols*1720+max(0,cols-1)*440)
        plan['downstream']=dict(x=operation_start,width=width,columns=max(1,cols),card_width=1720)
        right=operation_start+width
    return plan,max(9800,right+280)


class ComponentDiagram(Diagram):
    def __init__(self, spec, meta):
        generic_views=spec.get('layout')=='component_views'
        plan,width=component_view_plan(spec) if generic_views else (None,spec.get('canvas_width',9800))
        super().__init__(width, 20000, spec['title'], '', meta)
        self.spec=spec
        self.component_plan=plan
        self.base,self.edges,self.nodes,self.labels=[],[],[],[]
        self.text_boxes,self.edge_boxes=[],[]
        self.bg,self.ink,self.muted='#fafbf8','#203344','#506577'
        self.colors.update(output='#a34877',completion='#a34877')
        self.members,self.relations,self.headers={},[],[]
        self.caption_boxes,self.port_counts,self.port_usage=[],{},{}
        four_views=spec.get('layout') in {'four_views','connected_views'}
        connected=spec.get('layout')=='connected_views'
        self.description=spec['summary']+(' 定义独立说明核心抽象；产生、变换、消费三块保留可确认的调用和对象流，未知内部机制按证据边界标出。' if generic_views else ' 定义为独立结构块，产生、变换、消费三块由真实调用与对象流连通，灰色辅助区继续跟踪消费产物。' if connected else ' 四个一级研究视角围绕同一 Jaxpr；各区独立阅读，消费产物的后续去向单独放在灰色辅助区。' if four_views else ' 每图一个核心概念；命名连线区分程序、执行、输出与完成，右侧保留四视角源码说明。')
        self.text(180,112,spec['title'],64,bold=True)
        self.text(184,180,spec['summary'],31,self.muted)
        self.navigation()
        legend=[({'transform':'对象变换','result':'结果 / 句柄返回'}.get(color,label),color)
                for label,color in LEGEND] if generic_views else LEGEND
        for i,(label,color) in enumerate(legend):
            x=185+i*1170
            dash=' stroke-dasharray="9 7"' if color in DASHED else ''
            self.labels.append(f'<path d="M{x},334h76" stroke="{self.colors[color]}" stroke-width="4"{dash}/>')
            self.text(x+95,345,label,28,self.muted)
        if generic_views:
            core=spec['nodes'][spec['core_id']]['title']
            reading=spec.get('view_reading',[
                '唯一核心：'+core+'。定义独立；右侧按产生、变换、消费组织可确认的对象关系。',
                '连线区分源码可见调用、可选分支与证据边界。编号只索引关系，交叉拱桥表示互不连接。'])
            if len(reading)!=2:raise ValueError('view_reading must contain two guide lines')
            for y,line in zip((418,471),reading):self.text(184,y,line,29 if y==418 else 28,self.muted)
            links=[('定义','#view_definition'),('产生','#view_production'),('变换','#view_transformation'),('消费','#view_consumption')]
            if 'downstream' in plan:links.append(('后续对象与外部边界','#downstream'))
            for i,(label,anchor) in enumerate(links):
                self.text(184+i*1500,548,'↗ '+label,29,self.colors['program'],href=anchor)
        elif four_views:
            self.text(184,418,'唯一核心是 Jaxpr。定义块说明抽象；右侧三个操作块沿实际调用与对象流连接，变换可以跳过。' if connected else '唯一核心是 Jaxpr。四个视角回答不同问题；它们不是必须依次经过的四个阶段。',29,self.muted)
            self.text(184,471,'产生 → 按需变换 → 消费，同时保留直接消费、Pallas 再绑定与 IR 回接支路。编号只索引关系；交叉拱桥表示互不连接。' if connected else '每个视角内标明输入、处理者与结果；跨区链接定位同一抽象。编号只索引关系，交叉拱桥表示互不连接。',28,self.muted)
            for x,label,anchor in [(184,'定义','#view_definition'),(1550,'产生','#view_production'),
                                   (2916,'变换','#view_transformation'),(4282,'消费','#view_consumption'),
                                   (5800,'消费结果的后续去向','#downstream')]:
                self.text(x,548,'↗ '+label,29,self.colors['program'],href=anchor)
        else:
            self.text(184,418,'填色与粗边框只用于核心概念。四视角由卡片标签标明；虚线结合颜色区分返回、观察、可选分支与待确认。',29,self.muted)
            self.text(184,471,'沿编号连线阅读调用和对象流转；交叉拱桥表示互不连接。右栏补充定义、可选操作和观察接口，底部索引可双向定位。',28,self.muted)

    def navigation(self):
        key=self.spec['key']
        links=[('返回 overview 对应模块','../overview/overview-software-stack-components-layered.svg#'+OVERVIEW_IDS[key])]
        for neighbor in NEIGHBORS[key]:
            links.append((LABELS[neighbor]+' 详细图',os.path.relpath(PATHS[neighbor],Path(self.spec['path']).parent)))
        if key=='jax':links.append(('图内 · Pallas 两层程序','#inner_jaxpr'))
        x=184
        for label,href in links:
            self.text(x,257,'↗ '+label,29,self.colors['program'],href=href)
            x+=self.measure('↗ '+label,29)[1].width+120

    def card_layout(self,key,w):
        n=self.spec['nodes'][key];size=43 if key==self.spec['core_id'] else 36
        title=self.wrap(n['title'],w-66,size)
        body=[line for row in n['rows'] for line in self.wrap(row,w-66,29)]
        links=[('源码 · '+name,self.ref(name)) for name in n['refs']]
        if n.get('link'):links.append(('进入相关图 / 接口说明 ↗',n['link']))
        if key in self.spec['details']:links.append(('定位主图核心概念 ↑','#'+self.spec['core_id']))
        h=180+len(title)*49+len(body)*42+len(links)*35
        return size,title,body,links,h

    def draw_card(self,key,x,y,w,section):
        n=self.spec['nodes'][key];size,title,body,links,h=self.card_layout(key,w)
        self.boxes[key]=(x,y,w,h);self.members[key]=section
        core=key==self.spec['core_id'];color=n['color'];stroke=self.colors[color]
        fill=({'data':'#e0f4ed','unknown':'#fff1d9'}.get(color,'#e6f0fd')) if core else '#ffffff'
        dash=' stroke-dasharray="9 7"' if n.get('uncertain') else ''
        self.nodes.append(f'<g id="{key}" data-module="{self.spec["key"]}" data-owner="{escape(n["owner"],quote=True)}" data-core="{str(core).lower()}"><title>{escape(n["title"])}</title><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="{4 if core else 2}"{dash}/></g>')
        self.text(x+30,y+40,('核心概念 · ' if core else '')+n['tag'],24,stroke,bold=True,owner=key)
        yy=y+98
        for line in title:self.text(x+30,yy,line,size,bold=True,owner=key);yy+=49
        yy+=10
        for line in body:self.text(x+30,yy,line,29,self.muted,owner=key);yy+=42
        yy=y+h-30-len(links)*35
        self.text(x+30,yy,'归属 · '+n['owner'],23,self.muted,owner=key)
        for label,href in links:
            yy+=35
            if self.measure(label,23)[1].width>w-60:raise ValueError('Source link exceeds card width: '+label)
            self.text(x+30,yy,label,23,stroke,href=href,owner=key)
        return h

    def frame(self,key,x,y,w,h,title,owner,external=False):
        fill='#f1f3f5' if external else '#edf3fb'
        self.base.append(f'<g id="section_{key}" data-owner="{escape(owner,quote=True)}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="27" fill="{fill}" stroke="#b9c9d7" stroke-width="2"/></g>')
        self.text(x+38,y+63,title,40,bold=True)
        self.text(x+40,y+112,owner+(' · 外部接口 / 上下游' if external else ' · 内部机制与对象'),27,self.muted)
        self.headers.append((x+20,y+10,w-40,128))

    def arrange(self):
        if self.spec.get('layout')=='component_views':
            self.arrange_component_views()
            return
        if self.spec.get('layout')=='connected_views':
            self.arrange_connected_views()
            return
        if self.spec.get('layout')=='four_views':
            self.arrange_four_views()
            return
        y=610
        for i,part in enumerate(self.spec['sections']):
            top=y;row_y=y+220
            for row in part['rows']:
                cols=[1] if len(row)==1 else [0,2] if len(row)==2 else list(range(3))
                heights=[self.draw_card(key,440+2160*col,row_y,1720,'main_'+str(i)) for col,key in zip(cols,row) if key]
                row_y+=max(heights,default=0)+300
            h=row_y-top-100
            self.frame('main_'+str(i),220,top,6400,h,part['title'],part['owner'],part['external'])
            y=top+h+230
        main_bottom=y-230;top=610;side_y=top+220
        for key in self.spec['details']:side_y+=self.draw_card(key,7260,side_y,2140,'details')+70
        if self.spec['details']:self.frame('details',7120,top,2460,side_y-top+35,'四视角 · 补充定义与接口','按卡片归属阅读')
        self.body_bottom=max(main_bottom,side_y+35)
        self.h=self.body_bottom+600+math.ceil(len(self.spec['edges'])/3)*160

    def arrange_component_views(self):
        """Adapt the connected four-block style to each component's content."""
        parts={part['id']:part for part in self.spec['sections']}
        self.section_bounds={}

        def panel(ident,top):
            part=parts[ident];layout=self.component_plan[ident]
            x,w,cols,card_width=(layout[k] for k in ('x','width','columns','card_width'))
            auxiliary=part.get('auxiliary',False)
            title_size=36 if auxiliary else 44
            title=self.wrap(part['title'],w-80,title_size)
            question=self.wrap(part.get('question',''),w-80,28)
            owner=self.wrap('归属 · '+part['owner'],w-80,25)
            yy=top+63
            for line in title:self.text(x+38,yy,line,title_size,bold=True);yy+=54
            yy+=1
            for line in question:self.text(x+40,yy,line,28,self.muted);yy+=38
            yy+=15
            for line in owner:self.text(x+40,yy,line,25,self.muted);yy+=35
            header_bottom=yy+10;row_y=header_bottom+85
            for row in part['rows']:
                pitch=(w-440-card_width)/(cols-1) if cols>1 else 0
                if len(row)==cols:slots=list(range(cols))
                elif len(row)==1:slots=[(cols-1)/2]
                elif len(row)==2:slots=[0,cols-1]
                else:slots=[i*(cols-1)/(len(row)-1) for i in range(len(row))]
                heights=[self.draw_card(key,x+220+slot*pitch,row_y,card_width,ident)
                         for slot,key in zip(slots,row) if key]
                row_y+=max(heights,default=0)+320
            bottom=max(row_y-100,header_bottom+220)
            fill='#f2f3f3' if auxiliary else '#edf3fb'
            self.base.append(f'<g id="{ident}" data-view="{"auxiliary" if auxiliary else ident.removeprefix("view_")}" data-owner="{escape(part["owner"],quote=True)}"><rect x="{x}" y="{top}" width="{w}" height="{bottom-top}" rx="27" fill="{fill}" stroke="#b9c9d7" stroke-width="2"/></g>')
            self.headers.append((x+20,top+10,w-40,header_bottom-top))
            self.section_bounds[ident]=(x,top,w,bottom-top)
            return bottom

        definition_bottom=panel('view_definition',650)
        direct_count=sum(e.get('channel')=='direct' for e in self.spec['edges'])
        operation_top=850+max(0,direct_count-3)*70
        operation_bottom=max(panel(ident,operation_top) for ident in VIEW_IDS[1:])
        auxiliary_bottom=panel('downstream',operation_bottom+430) if 'downstream' in parts else 0
        self.body_bottom=max(definition_bottom,operation_bottom,auxiliary_bottom)
        self.h=self.body_bottom+600+math.ceil(len(self.spec['edges'])/3)*160

    def arrange_connected_views(self):
        """A static definition block beside three connected operation blocks."""
        sections={part['id']:part for part in self.spec['sections']}
        self.section_bounds={}

        def panel(ident,x,top,w,cols):
            part=sections[ident];row_y=top+290
            auxiliary=part.get('auxiliary',False)
            for row in part['rows']:
                card_width=2340 if ident=='view_definition' else 3100 if auxiliary else 1720
                pitch=(w-440-card_width)/(cols-1) if cols>1 else 0
                slots=list(range(len(row))) if len(row)==cols else [1] if len(row)==1 and cols==3 else [0,2] if len(row)==2 and cols==3 else list(range(len(row)))
                heights=[self.draw_card(key,x+220+slot*pitch,row_y,card_width,ident)
                         for slot,key in zip(slots,row) if key]
                row_y+=max(heights,default=0)+(240 if ident=='view_definition' else 320)
            bottom=row_y+(30 if ident=='view_definition' else -100)
            fill='#f2f3f3' if auxiliary else '#edf3fb'
            self.base.append(f'<g id="{ident}" data-view="{"auxiliary" if auxiliary else ident.removeprefix("view_")}" data-owner="{escape(part["owner"],quote=True)}"><rect x="{x}" y="{top}" width="{w}" height="{bottom-top}" rx="27" fill="{fill}" stroke="#b9c9d7" stroke-width="2"/></g>')
            self.text(x+38,top+63,part['title'],36 if auxiliary else 44,bold=True)
            question=self.wrap(part['question'],w-80,28)
            for i,line in enumerate(question):self.text(x+40,top+118+i*38,line,28,self.muted)
            self.text(x+40,top+171+(len(question)-1)*38,'归属 · '+part['owner'],25,self.muted)
            self.headers.append((x+20,top+10,w-40,205))
            self.section_bounds[ident]=(x,top,w,bottom-top)
            return bottom

        definition_bottom=panel('view_definition',220,650,2900,1)
        operation_bottom=max(panel(ident,x,850,6300,3) for ident,x in
            [('view_production',3700),('view_transformation',10400),('view_consumption',17100)])
        downstream_bottom=panel('downstream',3700,operation_bottom+430,19700,5)
        self.body_bottom=max(definition_bottom,downstream_bottom)
        self.h=self.body_bottom+600+math.ceil(len(self.spec['edges'])/6)*160

    def arrange_four_views(self):
        """Keep JAX's four research views explicit, with local companion cards.

        The lighter downstream panel is an auxiliary account of other objects,
        not a fifth view of Jaxpr. This layout is never used by other components.
        """
        top=650
        for part in self.spec['sections']:
            ident=part['id'];row_y=top+255
            for row in part['rows']:
                cols=[1] if len(row)==1 else [0,2] if len(row)==2 else list(range(3))
                heights=[self.draw_card(key,440+2160*col,row_y,1720,ident)
                         for col,key in zip(cols,row) if key]
                row_y+=max(heights,default=0)+300
            side_y=top+255
            for key in part.get('details',[]):
                side_y+=self.draw_card(key,7300,side_y,2060,ident+'_details')+80
            bottom=max(row_y-100,side_y+70)
            auxiliary=part.get('auxiliary',False)
            fill='#f2f3f3' if auxiliary else '#edf3fb'
            stroke='#cbd0d3' if auxiliary else '#b9c9d7'
            self.base.append(f'<g id="{ident}" data-view="{"auxiliary" if auxiliary else ident.removeprefix("view_")}" data-owner="{escape(part["owner"],quote=True)}"><rect x="220" y="{top}" width="9360" height="{bottom-top}" rx="27" fill="{fill}" stroke="{stroke}" stroke-width="2"/></g>')
            self.text(258,top+63,part['title'],36 if auxiliary else 44,bold=True)
            self.text(260,top+118,part['question'],28,self.muted)
            self.text(260,top+165,'归属 · '+part['owner'],25,self.muted)
            self.headers.append((240,top+10,9320,182))
            top=bottom+230
        self.body_bottom=top-230
        self.h=self.body_bottom+600+math.ceil(len(self.spec['edges'])/3)*160

    def sides(self,a,b):
        x,y,w,h=self.boxes[a];u,v,s,t=self.boxes[b]
        if self.spec.get('layout') in {'connected_views','component_views'}:
            operation_views={'view_production','view_transformation','view_consumption'}
            if self.members[a]!=self.members[b] and {self.members[a],self.members[b]}<=operation_views:
                return ('right','left') if x<u else ('left','right')
        if abs(y-v)<100:return ('right','left') if x<u else ('left','right')
        return ('bottom','top') if y<v else ('top','bottom')

    def port(self,key,side):
        x,y,w,h=self.boxes[key];index=self.port_usage.get((key,side),0)
        self.port_usage[key,side]=index+1;count=self.port_counts[key,side]
        fraction=.15+.7*(index+1)/(count+1)
        if side=='left':return (x,y+h*fraction),(-1,0)
        if side=='right':return (x+w,y+h*fraction),(1,0)
        if side=='top':return (x+w*fraction,y),(0,-1)
        return (x+w*fraction,y+h),(0,1)

    def caption(self,points,label,kind):
        size=26
        segments=sorted(zip(points,points[1:]),key=lambda e:(e[0][1]!=e[1][1],-abs(e[0][0]-e[1][0])-abs(e[0][1]-e[1][1])))
        obstacles=list(self.boxes.values())+self.headers+self.caption_boxes+getattr(self,'caption_port_boxes',[])
        for max_width in (660,380,200,120):
            lines=self.wrap(label,max_width,size)
            width=max(self.measure(line,size)[1].width for line in lines)
            height=42+37*(len(lines)-1)
            for a,b in segments:
                for fraction in (.5,.25,.75,.12,.88):
                    mx=a[0]+(b[0]-a[0])*fraction;my=a[1]+(b[1]-a[1])*fraction
                    options=[(mx-width/2,my-height+24),(mx-width/2,my+49)] if a[1]==b[1] else [(mx+30,my),(mx-width-30,my)]
                    for x,y in options:
                        rect=(x-10,y-32,width+20,height)
                        caption_right=self.w-60 if self.spec.get('layout') in {'connected_views','component_views'} else 6900
                        if x<60 or x+width>caption_right or y<550 or y+height>self.body_bottom:continue
                        if any(overlaps(rect,r,8) for r in obstacles):continue
                        if any(overlaps(rect,r[:4],7) for r in self.text_boxes):continue
                        for i,line in enumerate(lines):self.label(x,y+i*37,line,kind,size=size)
                        self.caption_boxes.append(rect)
                        return True
        return False

    def flows(self):
        for e in self.spec['edges']:
            for key,side in zip((e['source'],e['target']),self.sides(e['source'],e['target'])):
                self.port_counts[key,side]=self.port_counts.get((key,side),0)+1
        main_boxes=[box for key,box in self.boxes.items() if self.members[key]!='details']
        endpoints=[]
        for e in self.spec['edges']:
            a,b=e['source'],e['target'];sides=self.sides(a,b)
            p,pv=self.port(a,sides[0]);q,qv=self.port(b,sides[1])
            endpoints.append((p,(p[0]+40*pv[0],p[1]+40*pv[1]),(q[0]+40*qv[0],q[1]+40*qv[1]),q))
        if self.spec.get('avoid_endpoint_captions') or self.spec.get('layout')=='component_views':
            self.caption_port_boxes=[(min(u[0],v[0])-28,min(u[1],v[1])-28,
                                      abs(u[0]-v[0])+56,abs(u[1]-v[1])+56)
                                     for p,start,end,q in endpoints for u,v in ((p,start),(end,q))]
        for i,e in enumerate(self.spec['edges']):
            a,b=e['source'],e['target'];kind=e['kind'];code=f'{i+1:02d}'
            p,start,end,q=endpoints[i]
            occupied=[pair for route in self.edge_boxes for pair in zip(route,route[1:])]
            reserved=[segment for j,(u,v,w,z) in enumerate(endpoints) if j!=i for segment in ((u,v),(w,z))]
            try:
                bounds=(60,550,self.w-120,self.body_bottom-530) if self.spec.get('layout') in {'connected_views','component_views'} else (60,550,6860,self.body_bottom-530)
                if e.get('channel')=='direct':
                    if self.spec.get('layout')=='component_views':
                        px,py,pw,ph=self.section_bounds['view_production']
                        cx,cy,cw,ch=self.section_bounds['view_consumption']
                        direct_count=sum(item.get('channel')=='direct' for item in self.spec['edges'])
                        direct_index=sum(item.get('channel')=='direct' for item in self.spec['edges'][:i])
                        offset=100+300*(direct_index+1)/(direct_count+1)
                        track_y=min(py,cy)-115-direct_index*70
                        via=[(px+pw+offset,track_y),(cx-offset,track_y)]
                    else:via=[(10200,735),(16900,735)]
                    route=[]
                    stops=[start,*via,end]
                    for u,v in zip(stops,stops[1:]):
                        leg=orthogonal_route(u,v,bounds,main_boxes+self.headers+self.caption_boxes,occupied+reserved)
                        route.extend(leg if not route else leg[1:])
                else:
                    route=orthogonal_route(start,end,bounds,main_boxes+self.headers+self.caption_boxes,occupied+reserved)
            except ValueError as error:
                raise ValueError(f'{self.spec["key"]} relation {code} {a} → {b}: {error}') from error
            points=compact([p]+route+[q])
            self.edge(points,kind,dashed=kind in DASHED or e['optional'],bridge_over=occupied,width=3.5)
            if not self.caption(points,code+' · '+e['label'],kind):
                raise ValueError(f'No readable flow caption: {self.spec["key"]} {a} → {b}: {e["label"]}')
            self.relations.append(dict(e,code=code,points=points))

    def footer(self):
        y=self.body_bottom+125
        connected=self.spec.get('layout') in {'connected_views','component_views'}
        columns=max(3,min(6,int((self.w-380)//3190))) if self.spec.get('layout')=='component_views' else 6 if connected else 3
        pitch=(self.w-380)/columns if connected else 3190
        for i,line in enumerate(self.wrap(self.spec['foot'],self.w-500 if connected else 9300,28)):self.text(190,y+i*42,line,28,self.muted)
        y+=150;self.text(190,y,'关系索引 · 点击两端定位节点',38,bold=True)
        for i,e in enumerate(self.relations):
            col,row=i%columns,i//columns;x=190+col*pitch;yy=y+85+row*160
            a,b=e['source'],e['target'];color=self.colors[e['kind']]
            for line,href in [(e['code']+' · '+self.spec['nodes'][a]['title'],'#'+a),('→ '+self.spec['nodes'][b]['title'],'#'+b)]:
                self.text(x,yy,line,26,color,href=href);yy+=38
            self.text(x,yy,e['label'],25,self.muted)
        foot_y=y+125+math.ceil(len(self.relations)/columns)*160
        pins=' · '.join(r+' '+pin[:12] for r,pin in self.meta['source_pins'].items())
        self.text(190,foot_y,'固定源码：'+pins,25,self.muted)
        self.text(190,foot_y+45,'证据：固定源码检查与 SVG 静态校验；未执行编译、CPU / GPU / TPU 计算、TPU 模拟或性能测试。',25,self.muted)
        self.h=foot_y+110

    def save(self,path):
        if self.spec['core_id'] not in self.boxes:raise ValueError('Missing unique core concept')
        if set(self.boxes)&{'title','desc',*self.colors}:raise ValueError('Node ID conflicts with an SVG definition')
        pairs={(e['source'],e['target']) for e in self.relations}
        for chain in self.spec['paths']:
            if any(pair not in pairs for pair in zip(chain,chain[1:])):raise ValueError('Broken required workflow: '+str(chain))
        segments=[(i,a,b) for i,route in enumerate(self.edge_boxes) for a,b in zip(route,route[1:])]
        for index,(i,a,b) in enumerate(segments):
            if a[0]!=b[0] and a[1]!=b[1]:raise ValueError('Non-orthogonal route')
            for j,u,v in segments[index+1:]:
                if i==j:continue
                for axis in (0,1):
                    along=1-axis
                    if a[axis]==b[axis]==u[axis]==v[axis] and min(max(a[along],b[along]),max(u[along],v[along]))-max(min(a[along],b[along]),min(u[along],v[along]))>1:
                        raise ValueError(f'Different flows share a segment: {i} {j}: {a} {b} / {u} {v}')
        self.meta.update(component=self.spec['key'],core_concepts={self.spec['key']:self.spec['core_id']},
                         nodes={k:dict(title=n['title'],owner=n['owner'],view=n['tag']) for k,n in self.spec['nodes'].items()},
                         relations=self.relations,checked_paths=self.spec['paths'],
                         layout=('overview-style static component definition and three connected operation views'
                                 if self.spec.get('layout')=='component_views' else 'overview-style static Jaxpr definition and three connected operation views'
                                 if self.spec.get('layout')=='connected_views' else 'overview-style four primary Jaxpr research views with auxiliary downstream objects'
                                 if self.spec.get('layout')=='four_views' else 'overview-style component workflow with four-view references'))
        if self.spec.get('layout') in {'four_views','connected_views','component_views'}:
            self.meta['research_views']={part['id']:dict(title=part['title'],
                nodes=[key for row in part['rows'] for key in row if key]+part.get('details',[]))
                for part in self.spec['sections'] if not part.get('auxiliary')}
            self.meta['auxiliary_sections']=[part['id'] for part in self.spec['sections'] if part.get('auxiliary')]
        if self.spec.get('layout')=='connected_views':
            for e in self.relations:
                if 'view_definition' in (self.members[e['source']],self.members[e['target']]):
                    if e['kind']!='neutral' or self.members[e['source']]!=self.members[e['target']]:
                        raise ValueError('Definition must remain a separate structural view')
            required={('produced','consume_input'),('produced','transform_input'),
                      ('after','consume_input'),('mosaic_module','payload'),
                      ('outer_module','r3'),('input_arrays','execute')}
            if not required<=pairs:raise ValueError('Missing required cross-view object flow')
        if self.spec.get('layout')=='component_views':
            if self.members[self.spec['core_id']]!='view_definition':
                raise ValueError('The unique core belongs in the static definition view')
            if set(self.boxes)!=set(self.spec['nodes']):
                raise ValueError('Every component node must be assigned to a view')
            for e in self.relations:
                if 'view_definition' in (self.members[e['source']],self.members[e['target']]):
                    if e['kind'] not in {'neutral','unknown'} or self.members[e['source']]!=self.members[e['target']]:
                        raise ValueError('Definition must remain a separate structural view')
        super().save(path)
        root=ET.parse(path).getroot();ns={'s':'http://www.w3.org/2000/svg'}
        ids=[e.get('id') for e in root.iter() if e.get('id')]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate SVG IDs')
        if sum(e.get('data-core')=='true' for e in root.iter())!=1:raise ValueError('Expected one core concept')
        for a in root.findall('.//s:a',ns):
            link=a.get('href','')
            if link.startswith('#') and link[1:] not in ids:raise ValueError('Broken internal link: '+link)


def render(spec,metadata):
    d=ComponentDiagram(spec,metadata);d.arrange();d.flows();d.footer();return d


def main(argv=None, *, only=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--fetch-sources',action='store_true')
    parser.add_argument('--component',choices=list(PATHS),default=only)
    args=parser.parse_args(argv)
    specs=[s for s in diagrams() if not args.component or s['key']==args.component]
    for spec in specs:
        names={ref for n in spec['nodes'].values() for ref in n['refs']}
        meta=source_metadata(args.root,names,args.fetch_sources)
        output=args.output_dir or args.root/'research/software-stack'
        path=output/(Path(spec['path']).name if only and args.output_dir else spec['path'])
        path.parent.mkdir(parents=True,exist_ok=True)
        render(spec,meta).save(path)
        print(f'{spec["key"]}: {len(names)} verified anchors; one core; {len(spec["edges"])} named relationships')


if __name__=='__main__':main()
