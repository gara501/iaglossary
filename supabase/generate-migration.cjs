// Generate SQL migration from the TypeScript data files
// Strategy: Create temp CJS copies that can be required directly
const fs = require('fs');
const path = require('path');

function loadTsDataFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf-8');
    // Remove BOM
    content = content.replace(/^\uFEFF/, '');
    // Remove \r
    content = content.replace(/\r/g, '');
    // Remove import lines
    content = content.replace(/^import\s+.*$/gm, '');
    // Remove export default line
    content = content.replace(/^export\s+default\s+\w+;?\s*$/gm, '');
    // Remove TypeScript type annotations from const declarations
    content = content.replace(/const\s+(\w+)\s*:\s*\w+\[\]\s*=/g, 'const $1 =');
    // Write to temp file as CJS
    const tmpFile = filePath + '.tmp.cjs';
    fs.writeFileSync(tmpFile, content + '\nmodule.exports = ' +
        content.match(/const\s+(\w+)\s*=/)?.[1] + ';\n');
    try {
        const data = require(tmpFile);
        return data;
    } finally {
        fs.unlinkSync(tmpFile);
    }
}

function escapeSQL(str) {
    if (str === null || str === undefined) return 'NULL';
    return "'" + String(str).replace(/'/g, "''") + "'";
}

function arrayToSQL(arr) {
    if (!arr || arr.length === 0) return "'{}'";
    const items = arr.map(s => '"' + String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"').join(',');
    return "'{" + items + "}'";
}

const srcDir = path.join(__dirname, '..', 'src', 'data');

console.log('Loading data files...');
const glossaryEn = loadTsDataFile(path.join(srcDir, 'glossaryData.ts'));
const glossaryEs = loadTsDataFile(path.join(srcDir, 'glossaryDataEs.ts'));
const learningEn = loadTsDataFile(path.join(srcDir, 'learningData.ts'));
const learningEs = loadTsDataFile(path.join(srcDir, 'learningDataES.ts'));

let sql = `-- ============================================
-- Migration: Create glossary_terms and learning_items
-- Generated from local TypeScript data files
-- ============================================

-- 1. Create glossary_terms table
CREATE TABLE IF NOT EXISTS glossary_terms (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    slug TEXT NOT NULL,
    term TEXT NOT NULL,
    letter TEXT NOT NULL,
    summary TEXT NOT NULL,
    definition TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    related_terms TEXT[] DEFAULT '{}',
    lang TEXT NOT NULL CHECK (lang IN ('en', 'es')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(slug, lang)
);

-- 2. Create learning_items table
CREATE TABLE IF NOT EXISTS learning_items (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    creator TEXT NOT NULL,
    summary TEXT NOT NULL,
    link TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    lang TEXT NOT NULL CHECK (lang IN ('en', 'es')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(slug, lang)
);

-- 3. Enable Row Level Security
ALTER TABLE glossary_terms ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_items ENABLE ROW LEVEL SECURITY;

-- 4. RLS Policies: public read, authenticated write
CREATE POLICY "Public read glossary" ON glossary_terms
    FOR SELECT USING (true);

CREATE POLICY "Authenticated insert glossary" ON glossary_terms
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Authenticated update glossary" ON glossary_terms
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Authenticated delete glossary" ON glossary_terms
    FOR DELETE TO authenticated USING (true);

CREATE POLICY "Public read learning" ON learning_items
    FOR SELECT USING (true);

CREATE POLICY "Authenticated insert learning" ON learning_items
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Authenticated update learning" ON learning_items
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Authenticated delete learning" ON learning_items
    FOR DELETE TO authenticated USING (true);

-- 5. Auto-update updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER glossary_terms_updated_at
    BEFORE UPDATE ON glossary_terms
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER learning_items_updated_at
    BEFORE UPDATE ON learning_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- SEED DATA
-- ============================================

`;

function addGlossaryInserts(data, lang) {
    sql += `-- Glossary terms (${lang})\n`;
    for (const item of data) {
        sql += `INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (\n`;
        sql += `    ${escapeSQL(item.id)},\n`;
        sql += `    ${escapeSQL(item.term)},\n`;
        sql += `    ${escapeSQL(item.letter)},\n`;
        sql += `    ${escapeSQL(item.summary)},\n`;
        sql += `    ${escapeSQL(item.definition)},\n`;
        sql += `    ${escapeSQL(item.category)},\n`;
        sql += `    ${arrayToSQL(item.relatedTerms)},\n`;
        sql += `    ${escapeSQL(lang)}\n`;
        sql += `);\n\n`;
    }
}

function addLearningInserts(data, lang) {
    sql += `-- Learning items (${lang})\n`;
    for (const item of data) {
        sql += `INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (\n`;
        sql += `    ${escapeSQL(item.id)},\n`;
        sql += `    ${escapeSQL(item.title)},\n`;
        sql += `    ${escapeSQL(item.creator)},\n`;
        sql += `    ${escapeSQL(item.summary)},\n`;
        sql += `    ${escapeSQL(item.link)},\n`;
        sql += `    ${escapeSQL(item.category || '')},\n`;
        sql += `    ${escapeSQL(lang)}\n`;
        sql += `);\n\n`;
    }
}

addGlossaryInserts(glossaryEn, 'en');
addGlossaryInserts(glossaryEs, 'es');
addLearningInserts(learningEn, 'en');
addLearningInserts(learningEs, 'es');

const outPath = path.join(__dirname, 'migration.sql');
fs.writeFileSync(outPath, sql, 'utf-8');
console.log(`Migration written to ${outPath}`);
console.log(`Glossary EN: ${glossaryEn.length} terms`);
console.log(`Glossary ES: ${glossaryEs.length} terms`);
console.log(`Learning EN: ${learningEn.length} items`);
console.log(`Learning ES: ${learningEs.length} items`);
