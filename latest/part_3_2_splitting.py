#standard importing and chunk structure as before
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict
    
    def to_dict(self):
        return asdict(self)

class MathChunker:
    #used for latex identification and safe spliting
    LATEX_BLOCK = re.compile(r'\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]')
    LATEX_INLINE = re.compile(r'\$[^\$\n]+\$|\\\(.*?\\\)')
    LATEX_ENV = re.compile(r'\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}')
    #specific pattern for problems in olimpiad books
    OLYMPIAD_CODE = re.compile(
        r'^([A-Za-zА-Яа-яІіЇїЄєIVX]+\.\d{2}\.\d+\.?)\s+',
        re.MULTILINE
    )
    
    PROBLEM_START = re.compile(
        r'^(?:#{1,4}\s*)?\*{0,2}(?:Приклад|Задача|Example|Problem|ПРИКЛАД|ЗАДАЧА)\s*[№#]?\s*\d+\*{0,2}\.?',
        re.MULTILINE | re.IGNORECASE
    )
    #words that signify continuation of block. used to not to trigger next section context. should stick to existing chunk
    STICKY_KEYWORDS = [
        r"розв'?язання", r"відповідь", r"доповнення", r"вказівка",
        r"solution", r"proof", r"answer", r"remark", r"hint",
        r"зауваження", r"доведення", r"supplement",
        r"\d+\s*спосіб",
        r"спосіб\s*\d+",
        r"а чи знали ви",
    ]
    STICKY_PATTERN = re.compile(
        r'^\s*#{1,6}\s*\*{0,2}(' + '|'.join(STICKY_KEYWORDS) + r')',
        re.IGNORECASE | re.MULTILINE
    )
    #patterns for multiple choice answers, imperative instructions , and numbered lists
    ANSWER_OPTION = re.compile(r'^[а-яa-z]\)\s+', re.IGNORECASE | re.MULTILINE)
    
    INSTRUCTION_LINE = re.compile(
        r'^(.+(?:Обчислити|Знайти|Довести|Розв\'язати|Спростити|Визначити|'
        r'Calculate|Find|Prove|Solve|Simplify|Determine|Показати|Show|'
        r'Побудувати|Construct|Перетворити|Transform).+:|.+:)\s*$',
        re.MULTILINE | re.IGNORECASE
    )
    
    NUMBERED_ITEM = re.compile(
        r'^(\d+)[.\)]\s*(?:\([^)]*\)\s*)?',
        re.MULTILINE
    )
    
    TABLE_ROW = re.compile(r'^\|.+\|$', re.MULTILINE)

    #general markdown structure detection and metadata extraction
    HEADER = re.compile(
        r'^(#{1,6})\s+(.+?)\s*$',
        re.MULTILINE
    )
    
    THEOREM_START = re.compile(
        r'^((?:Теорема|Theorem|Лема|Lemma|Наслідок|Corollary|'
        r'Визначення|Definition|Твердження|Proposition|Аксіома|Axiom)\s*[\d.]*)',
        re.IGNORECASE | re.MULTILINE
    )
    
    GRADE_PATTERN = re.compile(r'(\d{1,2})\s*клас', re.IGNORECASE)
    YEAR_PATTERN = re.compile(r'\b((?:19|20)\d{2})\b')
    
    CONDITION_KEYWORDS = ['умов', 'завдання', 'задач', 'conditions', 'problems']
    SOLUTION_KEYWORDS = ['розв\'язання', 'відповід', 'solutions', 'answers', 'розв\'язання']
    #init
    def __init__(self, max_chunk_size=2000, min_chunk_size=150):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    #id based on hash
    def generate_chunk_id(self, content, source):
        hash_input = f"{source}:{content[:200]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    #finding latex and replacing with safe tokens  before splitting
    def protect_latex(self, text):
        placeholders = {}
        counter = [0]
        
        def replace(match):
            key = f"__LATEX_{counter[0]}__"
            placeholders[key] = match.group(0)
            counter[0] += 1
            return key
        
        protected = self.LATEX_ENV.sub(replace, text)
        protected = self.LATEX_BLOCK.sub(replace, protected)
        protected = self.LATEX_INLINE.sub(replace, protected)
        return protected, placeholders
    
    def restore_latex(self, text, placeholders):
        for key, value in placeholders.items():
            text = text.replace(key, value)
        return text
    
    #filtering noise. some \# are lists 
    def is_valid_header(self, line):
        line = line.strip()
        if not line.startswith('#'):
            return False
        header_text = re.sub(r'^#{1,6}\s*', '', line).strip().strip('*')
        if self.ANSWER_OPTION.match(header_text):
            return False
        if len(header_text) > 100:
            return False
        if '. ' in header_text and not header_text.endswith('.'):
            return False
        return True
    
    def is_sticky_header(self, header_text):
        text_lower = header_text.lower().strip().strip('*')
        for kw in self.STICKY_KEYWORDS:
            if re.search(kw, text_lower, re.IGNORECASE):
                return True
        return False
    
    #creating map structure
    def build_header_index(self, text):
        headers = []
        for match in self.HEADER.finditer(text):
            if not self.is_valid_header(match.group(0)):
                continue
            level = len(match.group(1))
            title = match.group(2).strip().strip('*')
            is_sticky = self.is_sticky_header(title)
            headers.append({
                'pos': match.start(),
                'end': match.end(),
                'level': level,
                'title': title,
                'is_sticky': is_sticky,
                'raw': match.group(0)
            })
        return sorted(headers, key=lambda x: x['pos'])
    
    def get_context_at_position(self, pos, headers):
        context_stack = {}
        section_type = 'theory'
        
        for h in headers:
            if h['pos'] >= pos:
                break
            level = h['level']
            title_lower = h['title'].lower()
            
            context_stack = {k: v for k, v in context_stack.items() if k < level}
            context_stack[level] = h['title']
            
            if any(kw in title_lower for kw in self.SOLUTION_KEYWORDS):
                section_type = 'blueprint'
            elif any(kw in title_lower for kw in self.CONDITION_KEYWORDS):
                section_type = 'drill'
        
        breadcrumbs = ' > '.join(
            context_stack[i] for i in sorted(context_stack.keys()) 
            if context_stack.get(i)
        )
        
        grade = None
        year = None
        for title in context_stack.values():
            if not grade:
                grade_match = self.GRADE_PATTERN.search(title)
                if grade_match:
                    grade = int(grade_match.group(1))
            if not year:
                year_match = self.YEAR_PATTERN.search(title)
                if year_match:
                    year = year_match.group(1)
        
        current_section = context_stack.get(max(context_stack.keys()), 'General') if context_stack else 'General'
        
        return {
            'breadcrumbs': breadcrumbs,
            'section_type': section_type,
            'grade': grade,
            'year': year,
            'current_section': current_section
        }
    
    # grabbing problem AND relevant solution
    def extract_problems_with_supplements(self, text, source, headers, placeholders):
        chunks = []
        remaining = text
        
        problem_matches = list(self.PROBLEM_START.finditer(text))
        if not problem_matches:
            return [], text
        
        extracted_ranges = []
        
        for i, match in enumerate(problem_matches):
            start = match.start()
            
            end = len(text)
            
            if i + 1 < len(problem_matches):
                end = min(end, problem_matches[i + 1].start())
            
            current_level = 99
            for h in headers:
                if h['pos'] <= start:
                    current_level = h['level']
                elif h['pos'] > start and not h['is_sticky']:
                    if h['level'] <= current_level + 1:
                        if 'доповнення' not in h['title'].lower():
                            end = min(end, h['pos'])
                            break
            
            content = text[start:end].strip()
            if len(content) < self.min_chunk_size:
                continue
            
            content = self.restore_latex(content, placeholders)
            
            ctx = self.get_context_at_position(start, headers)
            
            chunk_type = 'blueprint' if any(kw in content.lower() for kw in self.SOLUTION_KEYWORDS) else 'drill'
            
            chunks.append(Chunk(
                id=self.generate_chunk_id(content, source),
                content=content,
                metadata={
                    'source': source,
                    'type': chunk_type,
                    'strategy': 'problem_extraction',
                    'breadcrumbs': ctx['breadcrumbs'],
                    'grade': ctx['grade'],
                    'year': ctx['year'],
                    'has_solution': 'розв\'язання' in content.lower() or 'solution' in content.lower()
                }
            ))
            
            extracted_ranges.append((start, end))
        
        for start, end in reversed(extracted_ranges):
            remaining = remaining[:start] + '\n' + remaining[end:]
        
        return chunks, remaining
    
    #extractor for olimpiad notation (lack standard notation)
    def extract_olympiad_problems(self, text, source, headers, placeholders):
        chunks = []
        remaining = text
        
        codes = list(self.OLYMPIAD_CODE.finditer(text))
        if not codes:
            return [], text
        
        extracted_ranges = []
        
        for i, match in enumerate(codes):
            start = match.start()
            end = len(text)
            
            if i + 1 < len(codes):
                end = codes[i + 1].start()
            else:
                for h in headers:
                    if h['pos'] > start and not h['is_sticky'] and h['level'] <= 2:
                        end = min(end, h['pos'])
                        break
            
            content = text[start:end].strip()
            if not content:
                continue
            
            content = self.restore_latex(content, placeholders)
            ctx = self.get_context_at_position(start, headers)
            code = match.group(1)
            
            chunks.append(Chunk(
                id=self.generate_chunk_id(content, source),
                content=content,
                metadata={
                    'source': source,
                    'type': ctx['section_type'],
                    'strategy': 'olympiad_code',
                    'olympiad_code': code,
                    'breadcrumbs': ctx['breadcrumbs'],
                    'grade': ctx['grade'],
                    'year': ctx['year']
                }
            ))
            
            extracted_ranges.append((start, end))
        
        for start, end in reversed(extracted_ranges):
            remaining = remaining[:start] + '\n' + remaining[end:]
        
        return chunks, remaining
    #handling tables
    def anchor_tables(self, text):
        lines = text.split('\n')
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            if i + 1 < len(lines) and self.TABLE_ROW.match(lines[i + 1]):
                table_lines = [line]
                i += 1
                while i < len(lines) and (self.TABLE_ROW.match(lines[i]) or lines[i].strip().startswith('|')):
                    table_lines.append(lines[i])
                    i += 1
                result.append('<!-- TABLE_BLOCK -->')
                result.extend(table_lines)
                result.append('<!-- /TABLE_BLOCK -->')
            else:
                result.append(line)
                i += 1
        
        return '\n'.join(result)
    #drill processin
    def find_current_instruction(self, text, position):
        text_before = text[:position]
        matches = list(self.INSTRUCTION_LINE.finditer(text_before))
        if matches:
            return matches[-1].group(1).strip()
        return None
    
    def process_numbered_items(self, text, source, headers, placeholders, base_pos=0):
        chunks = []
        current_instruction = None
        
        items = list(self.NUMBERED_ITEM.finditer(text))
        if not items:
            return []
        
        for i, match in enumerate(items):
            new_instruction = self.find_current_instruction(text, match.start())
            if new_instruction:
                current_instruction = new_instruction
            
            ctx = self.get_context_at_position(base_pos + match.start(), headers)
            
            start = match.start()
            end = items[i + 1].start() if i + 1 < len(items) else len(text)
            item_content = text[start:end].strip()
            
            if len(item_content) < 20:
                continue
            
            item_content = self.restore_latex(item_content, placeholders)
            
            full_content = item_content
            if current_instruction:
                full_content = f"[Контекст: {current_instruction}]\n{item_content}"
            if ctx['breadcrumbs']:
                full_content = f"[{ctx['breadcrumbs']}]\n{full_content}"
            
            chunks.append(Chunk(
                id=self.generate_chunk_id(full_content, source),
                content=full_content,
                metadata={
                    'source': source,
                    'type': ctx['section_type'],
                    'section': ctx['current_section'],
                    'breadcrumbs': ctx['breadcrumbs'],
                    'grade': ctx['grade'],
                    'year': ctx['year'],
                    'has_instruction': current_instruction is not None,
                    'item_number': match.group(1),
                    'strategy': 'numbered_item'
                }
            ))
        
        return chunks
    
    def classify_content(self, text):
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in self.SOLUTION_KEYWORDS):
            return 'blueprint'
        
        if self.NUMBERED_ITEM.search(text) and not any(
            kw in text_lower for kw in self.SOLUTION_KEYWORDS
        ):
            return 'drill'
        
        if self.THEOREM_START.search(text):
            return 'theory'
        
        return 'theory'
    
    # main pipeline
    # 1. protect latex
    # 2. build index
    # 3. extract olimpiad
    # 4. exctract problems
    # 5. tables
    # 6. process remaining (drill and regular text)
    # 7. restore latex
    def chunk_document(self, text, source="unknown"):
        chunks = []
        
        protected_text, placeholders = self.protect_latex(text)
        
        headers = self.build_header_index(protected_text)
        
        olympiad_chunks, remaining = self.extract_olympiad_problems(
            protected_text, source, headers, placeholders
        )
        chunks.extend(olympiad_chunks)
        
        problem_chunks, remaining = self.extract_problems_with_supplements(
            remaining, source, headers, placeholders
        )
        chunks.extend(problem_chunks)
        
        remaining = self.anchor_tables(remaining)
        
        remaining_chunks = self.process_remaining(remaining, source, headers, placeholders)
        chunks.extend(remaining_chunks)
        
        for chunk in chunks:
            if '__LATEX_' in chunk.content:
                chunk.content = self.restore_latex(chunk.content, placeholders)
        
        return chunks
    
    def process_remaining(self, text, source, headers, placeholders):
        chunks = []
        
        split_headers = [h for h in headers if not h['is_sticky'] and h['level'] <= 3]
        
        if not split_headers:
            return self.chunk_by_size(text, source, headers, placeholders, 0)
        
        sections = []
        for i, header in enumerate(split_headers):
            start = header['pos']
            end = split_headers[i + 1]['pos'] if i + 1 < len(split_headers) else len(text)
            
            if start < len(text):
                sections.append({
                    'start': start,
                    'end': end,
                    'header': header
                })
        
        if split_headers and split_headers[0]['pos'] > 0:
            pre_content = text[:split_headers[0]['pos']]
            chunks.extend(self.chunk_by_size(pre_content, source, headers, placeholders, 0))
        
        for section in sections:
            section_text = text[section['start']:section['end']]
            title_lower = section['header']['title'].lower()
            
            is_drill = any(kw in title_lower for kw in 
                          self.CONDITION_KEYWORDS + ['вправ', 'exercise', 'problem', 'тест'])
            
            if is_drill or self.is_mostly_numbered(section_text):
                drill_chunks = self.process_numbered_items(
                    section_text, source, headers, placeholders, section['start']
                )
                if drill_chunks:
                    chunks.extend(drill_chunks)
                else:
                    chunks.extend(self.chunk_by_size(
                        section_text, source, headers, placeholders, section['start']
                    ))
            else:
                chunks.extend(self.chunk_by_size(
                    section_text, source, headers, placeholders, section['start']
                ))
        
        return chunks
    
    def chunk_by_size(self, text, source, headers, placeholders, base_pos):
        chunks = []
        
        text = re.sub(r'<!-- /?TABLE_BLOCK -->', '', text)
        text = text.strip()
        
        if len(text) < self.min_chunk_size:
            return []
        
        paragraphs = re.split(r'\n\s*\n+', text)
        current = ""
        current_start = base_pos
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current) + len(para) + 2 <= self.max_chunk_size:
                current += para + "\n\n"
            else:
                if current.strip() and len(current.strip()) >= self.min_chunk_size:
                    content = self.restore_latex(current.strip(), placeholders)
                    ctx = self.get_context_at_position(current_start, headers)
                    chunks.append(Chunk(
                        id=self.generate_chunk_id(content, source),
                        content=content,
                        metadata={
                            'source': source,
                            'type': self.classify_content(content),
                            'section': ctx['current_section'],
                            'breadcrumbs': ctx['breadcrumbs'],
                            'grade': ctx['grade'],
                            'year': ctx['year'],
                            'strategy': 'size_split'
                        }
                    ))
                current = para + "\n\n"
                current_start = base_pos
        
        if current.strip() and len(current.strip()) >= self.min_chunk_size:
            content = self.restore_latex(current.strip(), placeholders)
            ctx = self.get_context_at_position(current_start, headers)
            chunks.append(Chunk(
                id=self.generate_chunk_id(content, source),
                content=content,
                metadata={
                    'source': source,
                    'type': self.classify_content(content),
                    'section': ctx['current_section'],
                    'breadcrumbs': ctx['breadcrumbs'],
                    'grade': ctx['grade'],
                    'year': ctx['year'],
                    'strategy': 'size_split'
                }
            ))
        
        return chunks
    
    def is_mostly_numbered(self, text):
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return False
        numbered = sum(1 for l in lines if self.NUMBERED_ITEM.match(l.strip()))
        return numbered / len(lines) > 0.4


def process_all_markdown(input_dir, output_file,
                         max_chunk_size=2000,
                         min_chunk_size=150):
    chunker = MathChunker(max_chunk_size, min_chunk_size)
    all_chunks = []
    
    md_files = list(Path(input_dir).rglob("*.md"))
    print(f"found {len(md_files)} files")
    
    for md_path in md_files:
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            source = md_path.stem
            chunks = chunker.chunk_document(content, source=source)
            all_chunks.extend(chunks)
            print(f"  {source}: {len(chunks)} chunks")
        except Exception as e:
            print(f"error processing {md_path}: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([c.to_dict() for c in all_chunks], f, ensure_ascii=False, indent=2)
    
    types = {}
    grades = {}
    strategies = {}
    for c in all_chunks:
        t = c.metadata.get('type', 'unknown')
        types[t] = types.get(t, 0) + 1
        g = c.metadata.get('grade')
        if g:
            grades[g] = grades.get(g, 0) + 1
        s = c.metadata.get('strategy', 'unknown')
        strategies[s] = strategies.get(s, 0) + 1
    
    print(f"total: {len(all_chunks)} chunks")
    
    return all_chunks


if __name__ == "__main__":
    process_all_markdown(
        input_dir="parsed_output",
        output_file="chunks_v3.json",
        max_chunk_size=2000,
        min_chunk_size=150
    )