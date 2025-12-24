import os
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
SEMANTIC_AVAILABLE = True

# each chunk is one retrieved unit
@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict
    
    def to_dict(self):
        return asdict(self) # for json export

class MathChunker: # all chunking logic
    # regex
    # protect latex from being split or corrupted 
    LATEX_BLOCK = re.compile(r'\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]')
    LATEX_INLINE = re.compile(r'\$[^\$\n]+\$|\\\(.*?\\\)')
    #markdown headers
    HEADER = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    #detection of math structures
    THEOREM_PATTERNS = re.compile(
        r'(?:^|\n)((?:Теорема|Theorem|Лема|Lemma|Наслідок|Corollary|'
        r'Визначення|Definition|Приклад|Example|Задача|Problem|'
        r'Доведення|Proof|Розв\'язання|Solution|Зауваження|Remark|'
        r'Твердження|Proposition|Аксіома|Axiom)[\s.:№#\d]*)',
        re.IGNORECASE | re.MULTILINE
    )
    
    def __init__(self, embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedding_model_name = embedding_model
        self._embedding_model = None
    
    def embedding_model(self):
        if self._embedding_model is None and SEMANTIC_AVAILABLE:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def generate_chunk_id(self, content, source): #make unique id using hash
        hash_input = f"{source}:{content[:200]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    # latex protection from spliting, coruption, truncation, etc by placing instead placeholder
    def protect_latex(self, text):
        placeholders = {}
        counter = [0]
        
        def replace(match):
            key = f"__LATEX_{counter[0]}__"
            placeholders[key] = match.group(0)
            counter[0] += 1
            return key
        
        protected = self.LATEX_BLOCK.sub(replace, text)
        protected = self.LATEX_INLINE.sub(replace, protected)
        return protected, placeholders
    
    # removing placeholder, restore latex
    def restore_latex(self, text, placeholders):
        for key, value in placeholders.items():
            text = text.replace(key, value)
        return text
    
    # chunking
    
    # splits into math parts, avoids spliting wholesome segments
    def chunk_math_aware(self, text, max_chunk_size=1500, min_chunk_size=200, overlap_size=100, source="unknown"):
        chunks = []
        protected_text, placeholders = self.protect_latex(text)
        headers = list(self.HEADER.finditer(protected_text))
        segments = self.split_by_math_structures(protected_text)
        
        current_chunk = ""
        current_header = "Introduction"
        chunk_type = "general"
        
        for segment in segments:
            segment_type = self.classify_segment(segment)
            header_match = self.HEADER.search(segment)
            if header_match:
                current_header = header_match.group(2).strip()
            
            if len(current_chunk) + len(segment) > max_chunk_size and current_chunk:
                restored_content = self.restore_latex(current_chunk, placeholders)
                if len(restored_content.strip()) >= min_chunk_size:
                    chunks.append(Chunk(
                        id=self.generate_chunk_id(restored_content, source),
                        content=restored_content.strip(),
                        metadata={
                            "source": source,
                            "strategy": "math_aware",
                            "section": current_header,
                            "content_type": chunk_type,
                            "chunk_index": len(chunks),
                            "has_latex": "__LATEX_" in current_chunk
                        }
                    ))
                overlap = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else ""
                current_chunk = overlap + segment
                chunk_type = segment_type
            else:
                current_chunk += segment
                if segment_type != "general":
                    chunk_type = segment_type
        
        if current_chunk.strip():
            restored_content = self.restore_latex(current_chunk, placeholders)
            if len(restored_content.strip()) >= min_chunk_size:
                chunks.append(Chunk(
                    id=self.generate_chunk_id(restored_content, source),
                    content=restored_content.strip(),
                    metadata={
                        "source": source,
                        "strategy": "math_aware",
                        "section": current_header,
                        "content_type": chunk_type,
                        "chunk_index": len(chunks),
                        "has_latex": "__LATEX_" in current_chunk
                    }
                ))
        return chunks
    
    # splits into math structures. for function above
    def split_by_math_structures(self, text):
        boundaries = []
        for match in self.THEOREM_PATTERNS.finditer(text):
            boundaries.append(match.start())
        for match in self.HEADER.finditer(text):
            boundaries.append(match.start())
        
        boundaries = sorted(set(boundaries))
        if not boundaries:
            return re.split(r'(\n\s*\n)', text)
        
        segments = []
        prev = 0
        for boundary in boundaries:
            if boundary > prev:
                segments.append(text[prev:boundary])
            prev = boundary
        segments.append(text[prev:])
        return [s for s in segments if s.strip()]
    
    def classify_segment(self, segment):
        segment_lower = segment.lower()
        if any(kw in segment_lower for kw in ['теорема', 'theorem', 'лема', 'lemma']):
            return "theorem"
        elif any(kw in segment_lower for kw in ['доведення', 'proof']):
            return "proof"
        elif any(kw in segment_lower for kw in ['приклад', 'example']):
            return "example"
        elif any(kw in segment_lower for kw in ['задача', 'problem', 'вправа', 'exercise']):
            return "problem"
        elif any(kw in segment_lower for kw in ['розв\'язання', 'solution', 'відповідь', 'answer']):
            return "solution"
        elif any(kw in segment_lower for kw in ['визначення', 'definition']):
            return "definition"
        return "general"
    
    #semantic chunking to make more topic-coherent chunks
    def chunk_semantic(self, text, max_chunk_size=1500, similarity_threshold=0.5, source="unknown"):
        if not SEMANTIC_AVAILABLE:
            return self.chunk_math_aware(text, max_chunk_size, source=source)
        
        protected_text, placeholders = self.protect_latex(text)
        sentences = re.split(r'(?<=[.!?])\s+', protected_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return self.chunk_math_aware(text, max_chunk_size, source=source)
        
        embeddings = self.embedding_model.encode(sentences)
        chunks = []
        current_group = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            similarity = np.dot(current_embedding, embeddings[i]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
            )
            current_text = ' '.join(current_group)
            
            if similarity >= similarity_threshold and len(current_text) + len(sentences[i]) < max_chunk_size:
                current_group.append(sentences[i])
                current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
            else:
                chunk_content = self.restore_latex(' '.join(current_group), placeholders)
                chunks.append(Chunk(
                    id=self.generate_chunk_id(chunk_content, source),
                    content=chunk_content,
                    metadata={"source": source, "strategy": "semantic", "chunk_index": len(chunks), "sentence_count": len(current_group)}
                ))
                current_group = [sentences[i]]
                current_embedding = embeddings[i]
        
        if current_group:
            chunk_content = self.restore_latex(' '.join(current_group), placeholders)
            chunks.append(Chunk(
                id=self.generate_chunk_id(chunk_content, source),
                content=chunk_content,
                metadata={"source": source, "strategy": "semantic", "chunk_index": len(chunks), "sentence_count": len(current_group)}
            ))
        return chunks
    
    # hybrid approach
    # split by headers
    # if small - keep al
    # if large - perform split
    def chunk_hybrid(self, text, max_chunk_size=1500, min_chunk_size=200, source="unknown"):
        chunks = []
        protected_text, placeholders = self.protect_latex(text)
        headers = list(self.HEADER.finditer(protected_text))
        
        if not headers:
            return self.chunk_math_aware(text, max_chunk_size, min_chunk_size, source=source)
        
        sections = []
        for i, header in enumerate(headers):
            start = header.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(protected_text)
            sections.append({
                "header": header.group(2).strip(),
                "level": len(header.group(1)),
                "content": protected_text[start:end]
            })
        
        for section in sections:
            section_content = self.restore_latex(section["content"], placeholders)
            if len(section_content) <= max_chunk_size:
                if len(section_content) >= min_chunk_size:
                    chunks.append(Chunk(
                        id=self.generate_chunk_id(section_content, source),
                        content=section_content.strip(),
                        metadata={"source": source, "strategy": "hybrid", "section": section["header"], "header_level": section["level"], "chunk_index": len(chunks)}
                    ))
            else:
                sub_chunks = self.chunk_math_aware(section_content, max_chunk_size, min_chunk_size, source=source)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["parent_section"] = section["header"]
                    sub_chunk.metadata["strategy"] = "hybrid"
                    chunks.append(sub_chunk)
        return chunks
    
    def chunk_by_paragraphs(self, text, max_size, source):
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < max_size:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(Chunk(
                        id=self.generate_chunk_id(current, source),
                        content=current.strip(),
                        metadata={"source": source, "strategy": "paragraph"}
                    ))
                current = para + "\n\n"
        if current.strip():
            chunks.append(Chunk(
                id=self.generate_chunk_id(current, source),
                content=current.strip(),
                metadata={"source": source, "strategy": "paragraph"}
            ))
        return chunks

def process_all_markdown(input_dir, output_file, strategy="hybrid", **kwargs):
    chunker = MathChunker()
    all_chunks = []
    strategies = {
        "math_aware": chunker.chunk_math_aware,
        "semantic": chunker.chunk_semantic,
        "hybrid": chunker.chunk_hybrid
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    chunk_func = strategies[strategy]
    md_files = list(Path(input_dir).rglob("*.md"))
    
    for md_path in md_files:
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            source = md_path.stem
            chunks = chunk_func(content, source=source, **kwargs)
            all_chunks.extend(chunks)
        except Exception:
            pass
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([c.to_dict() for c in all_chunks], f, ensure_ascii=False, indent=2)
    return all_chunks


if __name__ == "__main__":
    INPUT_DIR = "parsed_output"
    OUTPUT_FILE = "chunks.json"
    process_all_markdown(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_FILE,
        strategy="hybrid",
        max_chunk_size=1500,
        min_chunk_size=200
    )