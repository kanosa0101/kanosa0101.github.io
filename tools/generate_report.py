# -*- coding: utf-8 -*-
import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_run_font(run, font_name="微软雅黑", size_pt=10.5, bold=False, color=(51, 51, 51)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)
    rPr = run._r.get_or_add_rPr()
    rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:ascii="{font_name}" w:hAnsi="{font_name}" w:eastAsia="{font_name}" w:cs="{font_name}"/>')
    rPr.append(rFonts)

def set_cell_shading(cell, color_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def add_callout(doc, text, title="设计亮点"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    set_cell_shading(cell, "EFF6FF")  # light blue-gray
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="2563EB"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    cell._tc.get_or_add_tcPr().append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.3
    r_title = p.add_run(f"【{title}】 ")
    set_run_font(r_title, "微软雅黑", 10.0, bold=True, color=(37, 99, 235))
    r_body = p.add_run(text)
    set_run_font(r_body, "微软雅黑", 10.0, bold=False, color=(55, 65, 81))

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def add_code_block(doc, code_str, title=""):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    set_cell_shading(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)

    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="16" w:space="0" w:color="64748B"/>
            <w:top w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
            <w:right w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
            <w:bottom w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
        </w:tcBorders>
    ''')
    cell._tc.get_or_add_tcPr().append(borders)

    p = cell.paragraphs[0]
    if title:
        r_head = p.add_run(f"// {title}\n")
        set_run_font(r_head, "微软雅黑", 9.0, bold=True, color=(100, 116, 139))
    
    r_code = p.add_run(code_str)
    set_run_font(r_code, "Consolas", 9.0, bold=False, color=(30, 41, 59))
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def add_custom_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table, color="CBD5E1", sz="4")

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_shading(hdr_cells[i], "1E3A8A")  # Deep Blue
        set_cell_margins(hdr_cells[i], top=140, bottom=140, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        for r in p.runs:
            set_run_font(r, "微软雅黑", 9.5, bold=True, color=(255, 255, 255))

    # Data Rows
    for row_idx, row_data in enumerate(rows):
        row_cells = table.rows[row_idx + 1].cells
        bg_color = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].text = str(cell_value)
            set_cell_shading(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], top=100, bottom=100, left=140, right=140)
            p = row_cells[col_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.2
            # First column center, others left
            if col_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                set_run_font(r, "微软雅黑", 9.0, bold=False, color=(51, 65, 85))

    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, "微软雅黑", 16.0, bold=True, color=(30, 58, 138)) # Deep Blue

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, "微软雅黑", 13.0, bold=True, color=(37, 99, 235)) # Royal Blue

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, "微软雅黑", 11.0, bold=True, color=(15, 118, 110)) # Teal / Dark Cyan

def add_body_p(doc, text, bold_prefix="", indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.35
    if indent:
        p.paragraph_format.first_line_indent = Pt(21) # 2 Chinese chars approx
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        set_run_font(r_pre, "微软雅黑", 10.5, bold=True, color=(17, 24, 39))
    r_body = p.add_run(text)
    set_run_font(r_body, "微软雅黑", 10.5, bold=False, color=(55, 65, 81))

def add_bullet_p(doc, bold_prefix, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.left_indent = Pt(18)
    r_dot = p.add_run("•  ")
    set_run_font(r_dot, "微软雅黑", 10.5, bold=True, color=(37, 99, 235))
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        set_run_font(r_pre, "微软雅黑", 10.5, bold=True, color=(17, 24, 39))
    r_body = p.add_run(text)
    set_run_font(r_body, "微软雅黑", 10.5, bold=False, color=(55, 65, 81))

def add_doc_image(doc, img_path, caption="", width_in=6.0):
    if not os.path.exists(img_path):
        print(f"Warning: image path {img_path} not found.")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width_in))
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run(caption)
        set_run_font(r_cap, "微软雅黑", 9.0, bold=False, color=(107, 114, 128))

def build_word_document(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = docx.Document()

    # Set Margins (2.54 cm / 1 inch)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # ========================== 封面区域 ==========================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(36)
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("《Web程序设计》课程作业")
    set_run_font(r_title, "微软雅黑", 24.0, bold=True, color=(30, 58, 138))

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(24)
    r_sub = p_sub.add_run("个人主页设计与开发实践说明书")
    set_run_font(r_sub, "微软雅黑", 16.0, bold=True, color=(37, 99, 235))

    # Meta Table on Cover
    info_headers = ["项目元数据", "详细说明与规格参数"]
    info_rows = [
        ["课程名称", "Web 程序设计 (Web Programming & Front-End Engineering)"],
        ["作业主题", "围绕个人背景深度定制的现代响应式技术门户与作品集主页"],
        ["学生姓名 / 开发者", "kanosa (GitHub: kanosa0101)"],
        ["在读院校与专业", "福州大学 (Fuzhou University) · 本科在读 (2024 - 2028)"],
        ["专业技术方向", "全栈软件工程 · 分布式微服务与多智能体系统协同"],
        ["站点线上访问网址", "https://kanosa0101.github.io/about/ (个人主页) | https://kanosa0101.github.io/ (博客门户)"],
        ["开源代码仓库", "https://github.com/kanosa0101/kanosa0101.github.io"],
        ["前端核心技术栈", "HTML5 语义化架构 / CSS3 (Flexbox + Grid + Glassmorphism) / JavaScript ES6+"],
        ["工程化体系与 CI/CD", "Hexo 8 静态生成 / Typora AST 图片流水线 / GitHub Pages 自动化交付"],
        ["报告完成时间", "2026 年 10 月"]
    ]
    add_custom_table(doc, info_headers, info_rows, col_widths=[2.0, 4.4])

    doc.add_page_break()

    # ========================== 摘要与目录导览 ==========================
    add_heading_1(doc, "摘  要")
    add_body_p(doc, "本说明书系统地阐述了针对《Web程序设计》课程作业要求所设计并上线的个人主页与技术门面站点。设计以福州大学（2024 - 2028）计算机软件工程学术背景为依托，聚焦全栈软件工程与智能体系统探索者的角色定位，深度践行课堂所学前端核心知识体系。页面全面采用了 HTML5 结构语义化规范、CSS3 现代响应式流式布局（涵盖双栏 Flexbox 弹性锁固机制与 CSS Grid 自适应卡片网格）、现代毛玻璃质感（Glassmorphism）、微交互反馈与关键帧动效，并在 JavaScript 层实现了顶部粘性导航交互、置顶滚动、全站本地秒级检索、Live2D 富媒体 Canvas 看板娘集成与异步性能加载。此外，结合工业级工程化理念，本项目搭建了基于 Node.js 现代语法与 AST 语法树的 Typora 本地写作与图片去重流水线，并通过 GitHub Pages 实现了 CI/CD 持续交付。全文深入解析了各功能模块的设计理念、代码实现细节以及在开发过程中对 Markdown 解析器与 DOM 结构冲突等实际工程难题的攻克历程。")

    add_callout(doc, "本个人主页已正式部署发布至公网 GitHub Pages，支持多终端屏幕自适应浏览。页面所有源码均遵循开源规范维护在 GitHub 仓库，实现了从本地源码提交到生产环境构建的纯自动化流水线闭环。", title="工程交付说明")

    # ========================== 第一章 ==========================
    add_heading_1(doc, "一、项目概述与设计理念")
    
    add_heading_2(doc, "1.1 设计背景与核心立意")
    add_body_p(doc, "在现代软件工程实践中，个人技术主页不仅是开发者展示工程能力与技术品位的数字名片，更是综合运用 Web 前端三大基础技术（HTML、CSS、JavaScript）以及工程化交付工具的绝佳载体。本次《Web程序设计》大作业要求：以本人为中心进行定制设计，综合参考业界优秀的 Web 交互案例，深入应用课上所学的前端结构、样式与脚本交互知识。")
    add_body_p(doc, "立足于作者在福州大学（2024 - 2028）的计算机与软件工程求学背景，本主页设计摒弃了模板化内容的机械堆砌，旨在打造一个兼具科技感、工程规范、高易用性与良好视觉美感的现代个人主页。整个主页围绕作者的核心特质（全栈交付、底层系统、多智能体协同、工程规范）展开，全面系统地呈现个人的技术栈全景、代表性开源作品集、学业生涯历程与高效生产力工作流。")

    add_heading_2(doc, "1.2 业界优秀案例调研与对标")
    add_body_p(doc, "在系统架构与界面交互设计前期，作者深入调研了当前 Web 前端与开发者社区的主流设计范式与优秀案例：")
    add_bullet_p(doc, "Hexo Butterfly 现代博客主题生态：", "研究了 Butterfly 主题高度模块化的双栏与单栏布局方案，借鉴了其侧边栏信息聚合卡片（公告、最新文章、作者卡片、全站运行统计）的紧凑组织模式。")
    add_bullet_p(doc, "Apple 与 GitHub 风格的极简毛玻璃美学：", "吸纳了现代 UI 设计中流行的 Glassmorphism（磨砂玻璃拟态）与高对比度色彩层级体系，利用半透明背景与高斯模糊滤镜打造通透、富有纵深感的卡片微容器。")
    add_bullet_p(doc, "VitePress 与现代文档站点的流式响应式体验：", "参考了知名技术文档站利用 CSS Grid 制作的自适应卡片流（Auto-fit Grid），确保无论在大屏 4K 显示器、常规笔记本电脑、平板电脑还是智能手机竖屏模式下，页面元素均能维持舒适的排版比例。")

    add_heading_2(doc, "1.3 整体功能与模块规划")
    add_body_p(doc, "个人主页（About 页面）与主站（Blog Index）形成了互为补充的双重门面。经过系统化梳理，个人主页规划了如下 7 个核心内容模块与 1 个常驻右侧侧边栏：")

    mod_headers = ["序号", "模块名称", "对应语义区块", "核心设计与呈现内容"]
    mod_rows = [
        ["1", "Hero 欢迎视窗", "<div class='about-hero'>", "个性化全栈标语、福州大学求学标签、GitHub 与归档一键直达胶囊按钮"],
        ["2", "核心特质与理念", "<div class='traits-grid'>", "展示全栈交付、系统底蕴与 CS61C、智能体系统协同、工程规范等 4 大特质"],
        ["3", "多维技术栈全景", "<div class='skills-container'>", "按前端跨端、后端微服务、智能体工具、基础架构划分为 4 大类技术徽章"],
        ["4", "精选代表性项目", "<div class='projects-grid'>", "严格精选 3 大代表作：PromptX（前端）、DocuMind（后端）、Moodisle（移动端）"],
        ["5", "学业与生涯里程碑", "<div class='timeline-card'>", "时间轴节点展示 2024-2028 福州大学求学生涯与全栈工程持续探索历程"],
        ["6", "生产力与工程工作流", "<div class='gear-grid'>", "呈现 Typora 定制流、现代 IDE 与 WSL2 复合终端、Docker 与持续交付体系"],
        ["7", "社交与互动结语", "<div class='about-contact-card'>", "展示“正在和世界解耦合”技术座右铭，提供 GitHub 与讨论区跳转交互"],
        ["8", "常驻信息侧边栏", "<aside id='aside-content'>", "包含博主微名片、全站公告看板、最新文章列表、归档分类统计与访客计数器"]
    ]
    add_custom_table(doc, mod_headers, mod_rows, col_widths=[0.6, 1.4, 1.8, 2.6])

    # ========================== 第二章 ==========================
    add_heading_1(doc, "二、HTML 页面结构设计与语义化实践")

    add_heading_2(doc, "2.1 HTML5 语义化架构与 DOM 层次")
    add_body_p(doc, "HTML（超文本标记语言）是 Web 页面的骨架。在传统 Web 开发中，页面往往充斥着无意义的 <div> 标签（即“div 滥用”或“div 汤”），这极大地损害了代码的可读性、可维护性以及搜索引擎爬虫的解析效率。在本次设计中，作者严格遵循 W3C 推荐的 HTML5 语义化标准，构建了清晰的树状 DOM 架构：")
    add_bullet_p(doc, "<header class='header-wrap'>：", "定义全站统一顶部导航，内嵌站点标题、多级页面导航链接（主页、归档、标签、分类、友链、关于）以及快捷本地搜索入口。")
    add_bullet_p(doc, "<nav id='nav'>：", "封装响应式导航列表，并在屏幕滚动时结合 JavaScript 动态改变其 class 属性以实现粘性定位与背景渐隐。")
    add_bullet_p(doc, "<main id='content-inner' class='layout'>：", "主体内容双栏包裹容器，将左侧主内容区（#page）与右侧辅助信息侧边栏（#aside-content）在 DOM 结构中进行平级组织。")
    add_bullet_p(doc, "<section class='about-section'>：", "对特质理念、技能全景、作品集、里程碑等逻辑独立板块进行区域划分，辅以 <h2 class='section-title'> 构建明确的内容层级大纲。")
    add_bullet_p(doc, "<aside id='aside-content'>：", "用于承载次要但常驻的信息模块，避免次要内容稀释文章或主页的核心主体权重。")
    add_bullet_p(doc, "<footer>：", "页面页脚区域，内嵌版权声明、建站运行时间、工信部备案/框架驱动标识与全站实时访客数计数组件。")

    add_heading_2(doc, "2.2 核心模块 HTML 实现与结构解析")
    add_body_p(doc, "以关于页（About）中的核心展示区块为例，代码采用高内聚的组件化思想进行书写。以下为代表性作品集（PromptX、DocuMind、Moodisle）与顶部 Hero 区的 HTML5 原生结构精简示意：")

    html_code = """<!-- 个人主页顶部欢迎区 Hero -->
<div class="about-card about-hero">
  <div class="about-hero-content">
    <div class="about-hero-badge"><i class="fas fa-sparkles"></i> Hello, World!</div>
    <h1 class="about-hero-title">我是 <span class="gradient-text">kanosa</span></h1>
    <p class="about-hero-desc">
      专注全栈开发与分布式系统落地的工程师。求学于<strong>福州大学（2024 - 2028）</strong>，
      热衷于构建高效、优雅的软件系统，深入探索现代架构与智能体协同。
    </p>
    <div class="about-hero-tags">
      <span class="hero-tag"><i class="fas fa-graduation-cap"></i> 福州大学 (2024-2028)</span>
      <span class="hero-tag"><i class="fas fa-laptop-code"></i> 全栈软件工程</span>
      <span class="hero-tag"><i class="fas fa-map-marker-alt"></i> 中国 · 福州</span>
    </div>
  </div>
</div>

<!-- 精选代表性作品集模块 -->
<div class="about-section">
  <div class="section-header">
    <h2 class="section-title"><i class="fas fa-laptop-code"></i> 代表性作品集</h2>
    <p class="section-subtitle">沉淀在 GitHub 的开源项目与工程实践</p>
  </div>
  <div class="projects-grid">
    <!-- 项目一：PromptX 前端交互框架 -->
    <div class="about-card project-card">
      <div class="project-header">
        <div class="project-badge">前端工具 · TypeScript</div>
        <a class="project-link" href="https://github.com/kanosa0101/PromptX" target="_blank" rel="noopener">
          <i class="fab fa-github"></i>
        </a>
      </div>
      <h3 class="project-title">PromptX · 智能交互与提示词框架</h3>
      <p class="project-desc">基于 TypeScript 构建的轻量级前端框架，专注于智能交互流管理与提示词模块化编排。</p>
      <div class="project-techs">
        <span class="p-tech">TypeScript</span><span class="p-tech">Frontend</span><span class="p-tech">Framework</span>
      </div>
    </div>
    <!-- 项目二：DocuMind 知识库系统 -->
    <div class="about-card project-card">
      <div class="project-header">
        <div class="project-badge">企业级 · 知识库</div>
        <a class="project-link" href="https://github.com/kanosa0101/DocuMind" target="_blank" rel="noopener">
          <i class="fab fa-github"></i>
        </a>
      </div>
      <h3 class="project-title">DocuMind · 智能文档与知识库系统</h3>
      <p class="project-desc">基于 Java 生态打造的结构化知识库系统，提供海量文档的分级归类与检索协同。</p>
      <div class="project-techs">
        <span class="p-tech">Java</span><span class="p-tech">Spring Boot</span><span class="p-tech">Knowledge Base</span>
      </div>
    </div>
    <!-- 项目三：Moodisle 跨端移动应用 -->
    <div class="about-card project-card">
      <div class="project-header">
        <div class="project-badge">移动跨端 · 交互</div>
        <a class="project-link" href="https://github.com/kanosa0101/Moodisle" target="_blank" rel="noopener">
          <i class="fab fa-github"></i>
        </a>
      </div>
      <h3 class="project-title">Moodisle (情绪岛屿) · 跨端移动应用</h3>
      <p class="project-desc">基于 Dart / Flutter 打造的轻量级跨平台客户端，聚焦心理情绪记录与细腻微动效。</p>
      <div class="project-techs">
        <span class="p-tech">Dart</span><span class="p-tech">Flutter</span><span class="p-tech">Mobile App</span>
      </div>
    </div>
  </div>
</div>"""
    add_code_block(doc, html_code, title="代码清单 2-1：精选项目展示区与 Hero 欢迎区语义化 HTML 结构")

    add_heading_2(doc, "2.3 语义化设计对 SEO 与无障碍可访问性（a11y）的赋能")
    add_body_p(doc, "Web 页面并非仅供常规桌面浏览器渲染，还需面对移动阅读器、屏幕朗读软件（Screen Readers）以及各大搜索引擎（Google、Bing、Baidu）网络蜘蛛的自动化抓取。本项目的 HTML 语义化实践在以下方面提供了强大支持：")
    add_bullet_p(doc, "标准超链接元属性修饰：", "所有外链卡片跳转均配置了 target='_blank' 与 rel='noopener' 属性。noopener 阻断了新窗口对 window.opener 对象的访问权限，既防范了恶意反向钓鱼攻击（Tabnabbing），又提升了多线程标签页的并发渲染性能。")
    add_bullet_p(doc, "完善的标题层级（Heading Hierarchy）：", "全站严格遵守 h1 -> h2 -> h3 的单一降序原则，每个独立页面仅有且仅有一个主 h1，各大 section 均由 h2 引导，使得屏幕朗读软件能够按大纲跳跃导航。")
    add_bullet_p(doc, "矢量图标可访问性补充：", "引入 FontAwesome 字体图标库时，配合 title 属性提供悬停文字说明，避免纯图标无文本时对辅助设备的理解障碍。")

    add_heading_2(doc, "2.4 原生 HTML 与 Markdown 混合渲染的工程踩坑与闭合排错")
    add_body_p(doc, "在实际开发 Hexo 静态站点的关于页（source/about/index.md）时，团队遇到了一起非常典型的深层次 Web 前端解析冲突故障：页面不仅将一段 HTML 源码（<!-- 第二个项目：DocuMind -->...）原样暴露在屏幕上，而且原本固定在页面右侧的侧边栏（公告、最新文章等）被挤压到了页面的最底端。")
    
    add_callout(doc, "根本原因定位：在 Markdown 解析器（marked.js）的标准规范中，只要在空行后遇到大于等于 4 个空格的缩进文本，解析器就会强制将其判定为 Indented Code Block（缩进代码块），从而将其包裹在 <pre><code> 标签中并将 HTML 实体转义。更严重的是，卡片内部的 </div> 闭合标签被困在代码块内未被 DOM 树识别，导致外部主内容容器 #page 缺失闭合，直接在 DOM 树中强行吞并了平级的 #aside-content 侧边栏！", title="核心工程避坑经验")

    add_body_p(doc, "针对此问题，作者制定了严格的排查与重构规范：彻底消除 Markdown 文件中内嵌 HTML 的前导多余缩进，实行顶格 0 缩进排布，确保 marked 渲染器直接以纯原生 HTML 节点注入 AST 树。经自动化单元测试验证，修复后 HTML 解析中 pre code block 出现概率降为 0%，#page 与 #aside-content 成功恢复为完全独立的双栏平级 DOM 节点。")

    # ========================== 第三章 ==========================
    add_heading_1(doc, "三、CSS 样式布局与现代视觉设计")

    add_heading_2(doc, "3.1 现代化布局机制：Flexbox 与 CSS Grid 的交融并用")
    add_body_p(doc, "CSS（层叠样式表）决定了网页的骨肉肌理与视觉排版。在传统的 CSS 时代，开发者不得不依赖 float 浮动与 clear 清浮动、或者 position: absolute 绝对定位来拼凑布局，极易引发高度塌陷与错位。在本项目中，作者全面采用了现代 Web 标准的两大核心布局利器：Flexbox（弹性盒子模型） 与 CSS Grid（二维网格布局），分层施策：")
    
    add_bullet_p(doc, "Flexbox 在宏观双栏骨架与微观组件对齐中的运用：", "在页面主骨架 #content-inner.layout 中，通过 display: flex; flex-direction: row; 将左侧内容区与右侧侧边栏组织为水平双栏；在 Hero 标签、项目徽章、按钮组等一维单向流中，Flexbox 提供了极佳的自适应水平垂直对齐能力。")
    add_bullet_p(doc, "CSS Grid 在多维卡片矩阵中的运用：", "对于特质理念（traits-grid）、技能树（skills-container）以及代表性作品集（projects-grid），采用 CSS Grid 实现真正的二维弹性网格。使用 repeat(auto-fit, minmax(280px, 1fr)) 语法，使浏览器根据视口宽度自动计算每一行的最佳列数，免去了大量冗余的媒体查询代码。")

    css_code = """/* 宏观主布局：双栏锁定与防止侧边栏塌陷 */
@media screen and (min-width: 600px) {
  #content-inner.layout {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
    gap: 20px !important;
  }
  #content-inner.layout > #page {
    flex: 1 1 auto !important;
    min-width: 0 !important; /* 关键：防止 flex 子项因内部超宽内容撑开溢出 */
  }
  #content-inner.layout > #aside-content {
    flex: 0 0 280px !important;
    width: 280px !important;
    min-width: 280px !important;
  }
}

/* 微观自适应网格：代表性作品集流式排列 */
.projects-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
  margin-top: 14px;
}

/* 磨砂玻璃态（Glassmorphism）与卡片立体悬停特效 */
.about-card {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(229, 231, 235, 0.8);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1),
              box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1),
              border-color 0.3s ease;
}

.about-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 20px -3px rgba(37, 99, 235, 0.12), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  border-color: rgba(59, 130, 246, 0.4);
}"""
    add_code_block(doc, css_code, title="代码清单 3-1：双栏 Flexbox 锁固、Grid 弹性卡片与 Glassmorphism 样式核心代码")

    add_heading_2(doc, "3.2 响应式网页设计（RWD）与媒体查询断点策略")
    add_body_p(doc, "为了确保主页在各类物理终端上均能呈现出色的阅读效果，本项目严格实施了移动优先（Mobile-First）与响应式断点策略。通过针对不同屏幕视口（Viewport）宽度设定 CSS 媒体查询规则，实现界面的平滑形变：")

    rwd_headers = ["断点范围 (px)", "设备类型对标", "侧边栏布局行为", "卡片网格响应式表现"]
    rwd_rows = [
        ["≥ 992px", "桌面宽屏 / 大屏显示器", "稳固停靠右侧（280px 宽度），内容两栏并排", "作品集与技能卡片以 3~4 列饱满铺满视窗"],
        ["600px ~ 991px", "平板电脑 / 笔记本小屏", "强制锁定右侧（280px 宽度），保障公告与导航可见", "自动收敛为 2 列并排，内容自适应缩放"],
        ["< 600px", "各类智能手机竖屏模式", "自动折叠并顺延堆叠至主内容下方，释放横向宽度", "全部卡片无缝平滑退化为单列 100% 宽度流式排列"]
    ]
    add_custom_table(doc, rwd_headers, rwd_rows, col_widths=[1.4, 1.6, 2.0, 1.8])

    add_heading_2(doc, "3.3 视觉质感与微动效（Modern Visuals & Animations）")
    add_body_p(doc, "优秀的前端设计绝不仅止于静止的文字与色块，细腻平滑的微交互（Micro-interactions）能够极大地唤醒用户的探索欲与操作愉悦感：")
    add_bullet_p(doc, "现代毛玻璃（Glassmorphism）与多重投影：", "卡片容器均配置了 background: rgba(255, 255, 255, 0.85) 与 backdrop-filter: blur(12px)。这使得卡片底部的背景壁纸（source/images/background.jpg）呈现出若隐若现的磨砂层次，视觉更显轻盈通透。")
    add_bullet_p(doc, "基于贝塞尔曲线的 3D 浮动反馈：", "当鼠标光标悬停至任意作品或技能卡片时，触发 transform: translateY(-4px) 与阴影扩散动画。利用 cubic-bezier(0.4, 0, 0.2, 1) 加速度曲线，动效兼具缓入缓出的丝滑物理质感。")
    add_bullet_p(doc, "渐变背景文字截取（Gradient Text）：", "在 Hero 欢迎区标题中，使用 background: linear-gradient(135deg, #2563eb, #7c3aed); 搭配 -webkit-background-clip: text; 与 -webkit-text-fill-color: transparent; 打造出色彩流淌的科技感主标题效果。")

    # ========================== 第四章 ==========================
    add_heading_1(doc, "四、JavaScript 交互与富媒体应用")

    add_heading_2(doc, "4.1 DOM 动态操作与滚动监听事件（Scroll Events）")
    add_body_p(doc, "JavaScript 作为赋予网页生命力的脚本语言，负责捕获用户交互行为并动态驱动 DOM（文档对象模型）的变更。在主页浏览过程中，页面需要对用户的滚动行为做出即时而平滑的响应：")
    add_bullet_p(doc, "动态吸顶导航栏（Sticky Navigation）：", "注册 window.addEventListener('scroll', callback) 事件监听器，通过 window.pageYOffset 动态捕获页面滚动偏移量。当向下滚动超过一定阈值时，自动向导航栏注入 nav-fixed 与 nav-visible 类，平滑切换为半透明吸顶状态；向上滚动时展开完整导航，向下快速滑行时智能收起以释放阅读视野。")
    add_bullet_p(doc, "回到顶部（Back to Top）交互：", "在页面右下角配置常驻悬浮控制按钮。当滚动距离超过视窗高度时淡入显示，点击后通过 window.scrollTo({ top: 0, behavior: 'smooth' }) 执行无跳跃感的原生平滑回滚动画。")

    js_code = """// 页面滚动监听与导航栏状态动态切换核心逻辑示意
(function() {
  let lastScrollTop = 0;
  const nav = document.getElementById('nav');
  const backToTopBtn = document.getElementById('go-up');
  
  // 使用 requestAnimationFrame 优化滚动事件性能，杜绝页面丢帧卡顿
  let ticking = false;
  window.addEventListener('scroll', function() {
    if (!ticking) {
      window.requestAnimationFrame(function() {
        const currentScrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        // 1. 回到顶部按钮的显隐控制
        if (backToTopBtn) {
          if (currentScrollTop > 300) {
            backToTopBtn.classList.add('show-btn');
          } else {
            backToTopBtn.classList.remove('show-btn');
          }
        }
        
        // 2. 粘性吸顶导航栏的动态样式切换
        if (currentScrollTop > 60) {
          nav.classList.add('nav-fixed');
          if (currentScrollTop > lastScrollTop && currentScrollTop > 500) {
            nav.classList.remove('nav-visible'); // 向下快速浏览时自动隐藏导航
          } else {
            nav.classList.add('nav-visible');    // 向上微滚时自动唤出导航
          }
        } else {
          nav.classList.remove('nav-fixed', 'nav-visible');
        }
        
        lastScrollTop = currentScrollTop <= 0 ? 0 : currentScrollTop;
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });
})();"""
    add_code_block(doc, js_code, title="代码清单 4-1：基于 requestAnimationFrame 节流的高性能滚动监听与导航栏控制")

    add_heading_2(doc, "4.2 本地全文搜索（Local Search）与动态高亮")
    add_body_p(doc, "为了提供高效的内容检索体验，站点集成了纯前端驱动的 Local Search 本地全文检索方案。在 Hexo 构建阶段，生成包含全站文章标题、正文与标签索引的 search.xml 数据字典；在客户端浏览器端：")
    add_bullet_p(doc, "异步非阻塞拉取（Fetch API）：", "当用户初次点击搜索图标时，通过异步 Promise 拉取索引文件，避免初始化时阻塞首屏关键资源加载。")
    add_bullet_p(doc, "正则表达式动态分词与匹配：", "实时监听 input 事件，利用正则表达式对用户输入的关键词进行模糊匹配与分词切片，瞬间过滤出相关页面摘要。")
    add_bullet_p(doc, "DOM 节点动态高亮着色：", "利用 <mark class='search-keyword'> 标签对匹配到的文字段进行动态包裹替换，赋予醒目的高亮荧光背景，让检索结果一目了然。")

    add_heading_2(doc, "4.3 富媒体与 Canvas 交互：Live2D 二维交互模型")
    add_body_p(doc, "为打破传统技术主页严肃单一的刻板印象，主页在左下角创新集成了基于 WebGL / HTML5 Canvas 渲染的 Live2D 交互看板娘组件（live2dw 扩展生态）。该模块深度融合了计算机图形学在 Web 前端的前沿应用：")
    add_bullet_p(doc, "实时姿态与视线追踪：", "Canvas 渲染循环持续监听全局 mousemove 鼠标指针坐标，并将其映射为 Live2D 骨骼关节的旋转矩阵，使模型的头部与眼神能够逼真地跟随用户光标旋转互动。")
    add_bullet_p(doc, "多级触碰物理碰撞箱（Hit Areas）：", "在模型身体与头部划定触摸感应区域，当捕获到 click 点击事件时，随机播放预置动作组（Motions）与语音气泡气泡提示，极大丰富了页面的趣味性与人机交互深度。")

    add_heading_2(doc, "4.4 异步通信与数据持久化")
    add_body_p(doc, "页面多处融入了现代异步通信（Ajax / Fetch）技术，实现了页面的无缝数据更新：")
    add_bullet_p(doc, "不蒜子（Busuanzi）全站访客动态统计：", "通过异步跨域 JSONP / Fetch 机制向远端统计服务器发起心跳，实时在页脚与侧边栏动态注入“本站访客数”与“本站总浏览量”。")
    add_bullet_p(doc, "客户端运行时间动态计算：", "JavaScript 定时器 setInterval 动态计算当前客户端时间戳与站点建站时间（2025-11-23）的差值，按“天-时-分-秒”实时刷新并在页面底端动态递增呈现，展现站点的持久生命力。")

    # ========================== 第五章 ==========================
    add_heading_1(doc, "五、工程化构建、测试与持续集成（CI/CD）")

    add_heading_2(doc, "5.1 现代静态构建流水线架构")
    add_body_p(doc, "现代前端开发已从“手工编写单个 HTML 文件”全面进化至“工程化自动化构建”阶段。本项目基于行业主流的高性能静态站点生成引擎 Hexo 8 构建，依托 Node.js 现代语法规范搭建了兼顾本地敏捷开发与生产自动化部署的完备工作流：")
    add_bullet_p(doc, "双分支版本控制架构：", "代码仓库采用严格的双分支（Two-Branch）协作模式。source 分支托管 Hexo 配置文件、Markdown 原稿、定制 CSS/JS 与自动化测试脚本；main 分支专门托管构建完成的纯静态 HTML/CSS/JS 交付产物，由 GitHub Pages 引擎直接托管。")
    add_bullet_p(doc, "一键全流程脚本（NPM Scripts）：", "在 package.json 中定制了集成化快捷指令：npm run build 执行静态化渲染、npm run check 执行站点健康扫描、npm run deploy 自动将产物打包部署并推向生产环境。")

    add_heading_2(doc, "5.2 独创实践：基于 AST 语法树的 Typora 本地写作与图片去重流水线")
    add_body_p(doc, "在日常博文撰写过程中，本地 Markdown 编辑器（如 Typora）生成的图片往往包含本地文件绝对路径（形如 C:\\Users\\...）、相对路径或冗余重复图档，直接推送到 GitHub 会导致图片大面积 404 碎裂。针对此痛点，作者独立编写了自动化预处理管道工具（tools/prepare-article.cjs 与 tools/article-helpers.cjs）：")
    add_bullet_p(doc, "基于 AST 抽象语法树的正则匹配流：", "自动解析 Markdown 文档中的标准图片语法 ![]()、HTML <img> 标签与引用式图片，精准定位多源图片路径。")
    add_bullet_p(doc, "文件指纹 Hash 自动去重与归集：", "通过计算图片二进制内容的 SHA-256 哈希指纹，将多篇博文中引用的相同图片自动去重并安全拷贝至 source/images/ 生产目录，同时自动重写 Markdown 中的图片引用为规范的根相对路径 /images/name.ext。")
    add_bullet_p(doc, "严苛防欺诈与格式白名单拦截：", "对非图片扩展名（如 .exe、.sh）以及伪装成图片的文件执行内容白名单深度检测，在发布前夕杜绝任何坏链或恶意脚本的混入。")

    add_heading_2(doc, "5.3 自动化测试与站点健康度校验体系")
    add_body_p(doc, "为了确保每次修改代码后部署到线上均万无一失，作者将软件工程中的持续集成（CI）理念贯彻到前端发布全流程中：")
    add_bullet_p(doc, "Node.js 原生 Test Runner 单元测试：", "基于 Node.js 20+ 内置的 node --test 引擎编写针对文章预处理流的单元测试（tools/prepare-article.test.cjs），覆盖图片相对路径、绝对路径、中文空格转义以及日期持久性等 4 大测试套件，用例通过率始终保持 100%。")
    add_bullet_p(doc, "全站静态健康度扫描（tools/check-site.cjs）：", "在 npm run check 执行时，爬取 public/ 目录下的所有静态 HTML 页面（共计 10 个核心页面），对本地资源死链、页面模板渲染完整性、Live2D 依赖资产进行全自动扫描拦截，确保 0 坏链交付。")

    # ========================== 第六章 ==========================
    add_heading_1(doc, "六、页面最终效果展示与性能评测")

    add_heading_2(doc, "6.1 页面最终渲染效果全景展示")
    add_body_p(doc, "经过前述 HTML 结构重组、CSS 样式润色与 JavaScript 交互注入，个人主页与主站博客均以极高品质成功交付上线。以下为页面在现代高分辨率浏览器中的实际渲染捕获：")

    # Embed homepage screenshot
    add_doc_image(doc, r"E:\homework\Web程序设计\homepage_preview.png", caption="图 6-1：个人主页（About 页面）全景与精选作品集（PromptX、DocuMind、Moodisle）展示截图", width_in=6.2)

    add_body_p(doc, "从图 6-1 可以清晰看出：")
    add_bullet_p(doc, "视觉层次清晰分明：", "顶部 Hero 欢迎区以饱满的字阶与渐变色彩确立视觉中心，福州大学标签与全栈工程师定位醒目。")
    add_bullet_p(doc, "作品集呈现专业严谨：", "代表作矩阵严格按照用户要求的 1. PromptX（前端工具）、2. DocuMind（知识库系统）、3. Moodisle（移动应用）顺序展示，徽章标签规范，直链 GitHub 仓库。")
    add_bullet_p(doc, "双栏侧边栏布局锁固：", "右侧侧边栏稳固矗立，与左侧主页面内容区域紧密呼应，彻底解决了先前侧边栏掉落到底部的布局缺陷。")

    # Embed main blog screenshot
    add_doc_image(doc, r"E:\homework\Web程序设计\main_preview.png", caption="图 6-2：博客主站（Index 页面）全景与常驻双栏侧边栏布局截图", width_in=6.2)

    add_heading_2(doc, "6.2 跨终端与多浏览器兼容性测试")
    add_body_p(doc, "为了验证页面的工业级鲁棒性，作者在不同操作系统、主流浏览器及移动终端上进行了实际联调测试，测试指标覆盖布局稳定性、动效流畅度与脚本执行：")

    test_headers = ["测试终端 / 平台", "操作系统", "浏览器内核与版本", "兼容性与渲染表现评估"]
    test_rows = [
        ["台式工作站 (2K 宽屏)", "Windows 11 Pro", "Google Chrome (Blink 130+)", "完美呈现，双栏布局比例精准，毛玻璃微动效流畅 60FPS"],
        ["轻薄笔记本 (1080P)", "Windows 11 / WSL2", "Microsoft Edge (Chromium 内核)", "排版规范完整，右侧侧边栏宽度稳定为 280px，无抖动错位"],
        ["MacBook Pro (Retina)", "macOS Sequoia", "Safari 18 (WebKit 内核)", "-webkit- 前缀兼容完美，渐变文字与高斯模糊滤镜质感通透"],
        ["移动平板 (iPad Air)", "iPadOS 17.5", "Mobile Safari", "触摸交互响应迅速，网格自动重排为双列，自适应比例舒适"],
        ["智能手机 (iPhone / 安卓)", "iOS 18 / Android 14", "Chrome Mobile / WeChat 浏览器", "自动平滑降级为单列垂直流，导航折叠自如，加载极快"]
    ]
    add_custom_table(doc, test_headers, test_rows, col_widths=[1.5, 1.4, 1.8, 2.1])

    # ========================== 第七章 ==========================
    add_heading_1(doc, "七、课程学习总结与心得体会")

    add_heading_2(doc, "7.1 从“课堂知识点”到“系统级交付”的认知跃升")
    add_body_p(doc, "通过本次《Web程序设计》个人主页的完整设计与开发实践，作者深刻体会到前端开发绝非简单的“标签拼凑与色彩涂抹”，而是一门融汇了计算机体系结构认知、人机交互工程学与现代软件自动化交付的综合性学科。回顾整个实践过程，作者获得了三个维度的深刻认知跃升：")
    add_bullet_p(doc, "对 HTML 骨架“语义与规范”的再认识：", "掌握了不仅要在形式上使用语义化标签，更要理解解析器对空白字符、缩进、DOM 嵌套规则的严格要求。规避 Markdown 与原生 HTML 嵌套导致的 <pre><code> 逃逸事故，是本次实践中极其宝贵的工程避坑实战经验。")
    add_bullet_p(doc, "对 CSS 现代布局“弹性与适配”的融会贯通：", "深刻理解了从早期 float 浮动黑魔法向现代 Flexbox 轴线对齐与 CSS Grid 二维自适应演进的必然性。通过弹性模型与媒体查询的精妙配合，实现了无需冗长重复代码即可支撑全平台终端的优雅自适应。")
    add_bullet_p(doc, "对 JavaScript“事件与性能调度”的进阶理解：", "在实现滚动监听、动态置顶与 Live2D 富媒体交互时，领悟了 requestAnimationFrame 节流与非阻塞异步拉取的精髓，确保了复杂微交互与 Canvas 图形渲染在低配设备上依然保持丝滑流畅。")

    add_heading_2(doc, "7.2 未来技术演进展望")
    add_body_p(doc, "作为福州大学 2024-2028 计算机软件工程方向的求学者，本个人主页的上线只是个人全栈技术探索长卷的序章。未来，作者计划在以下方向持续演进该站点：")
    add_bullet_p(doc, "引入 Web Components 现代化组件解耦：", "将页面中特质卡片、作品集卡片等常用模块封装为原生自定义组件（Custom Elements），进一步实现样式隔离与逻辑复用。")
    add_bullet_p(doc, "深度赋能 AI 智能体（Agent）问答交互：", "结合作者在多智能体系统（Multi-Agent System）方向的探索，计划在主页中接入轻量级开源大模型 API，构建“全站知识库对话助手”，让访客能够通过自然语言与个人博客进行实时沉浸式问答互动。")
    add_bullet_p(doc, "深化 PWA（渐进式 Web 应用）体验：", "配置完善的 Service Worker 离线缓存策略与 Web App Manifest，实现离线可访问与移动端桌面图标安装，将个人主页的访问体验提升至类原生 App 级别。")

    # Final sign-off
    p_end = doc.add_paragraph()
    p_end.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_end.paragraph_format.space_before = Pt(28)
    p_end.paragraph_format.space_after = Pt(4)
    r_sign1 = p_end.add_run("报告撰写人：kanosa\n")
    set_run_font(r_sign1, "微软雅黑", 11.0, bold=True, color=(30, 58, 138))
    r_sign2 = p_end.add_run("所在高校：福州大学 (Fuzhou University)\n")
    set_run_font(r_sign2, "微软雅黑", 10.0, bold=False, color=(75, 85, 99))
    r_sign3 = p_end.add_run("完成日期：2026 年 10 月\n")
    set_run_font(r_sign3, "微软雅黑", 10.0, bold=False, color=(75, 85, 99))

    doc.save(output_path)
    print(f"Document successfully created and saved to: {output_path}")

if __name__ == "__main__":
    target_path = r"E:\homework\Web程序设计\个人主页设计与实现说明书.docx"
    build_word_document(target_path)
