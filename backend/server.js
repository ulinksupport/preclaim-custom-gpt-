const express  = require('express');
const cors     = require('cors');
const multer   = require('multer');
const OpenAI   = require('openai');
const pdfParse = require('pdf-parse');
const fs       = require('fs');
const path     = require('path');

const app  = express();
const port = process.env.PORT || 3000;

// ── OpenAI client ──
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ── CORS & iframe headers (single source of truth) ──
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('X-Frame-Options', 'ALLOWALL');
  if (req.method === 'OPTIONS') return res.sendStatus(204); // preflight
  next();
});

// ── Parse JSON bodies ──
app.use(express.json({ limit: '50mb' }));

// ── File upload (memory storage — no disk writes needed) ──
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB max
  fileFilter: (req, file, cb) => {
    const allowed = ['application/pdf','image/jpeg','image/jpg','image/png','image/webp'];
    if (allowed.includes(file.mimetype)) cb(null, true);
    else cb(new Error('Only PDF, JPG, PNG, or WEBP files are supported.'));
  }
});

// ── Serve Static Frontend (index.html) ──
app.use(express.static(path.join(__dirname, '..')));

// ── Health check ──
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'Ulink Pre-Claim AI Backend' });
});

// ══════════════════════════════════════════════
//  POST /analyse  — main endpoint
// ══════════════════════════════════════════════
app.post('/analyse', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded.' });
  try {
    let result;
    if (req.file.mimetype === 'application/pdf') result = await analysePDF(req.file.buffer);
    else result = await analyseImage(req.file.buffer, req.file.mimetype);
    res.json(result);
  } catch (err) {
    console.error('Analysis error:', err.message);
    res.status(500).json({ error: err.message || 'AI analysis failed.' });
  }
});

// ── Proxy endpoint for pre-built content ──
app.post('/proxy-analyse', async (req, res) => {
  const { content } = req.body;
  if (!content) return res.status(400).json({ error: 'No content provided.' });
  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      temperature: 0.2,
      messages: [{ role: 'system', content: SYSTEM_PROMPT }, { role: 'user', content }]
    });
    res.json(parseAIResponse(completion.choices[0].message.content));
  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Proxy endpoint to amend existing results ──
app.post('/proxy-amend', async (req, res) => {
  const { currentData, instruction } = req.body;
  if (!currentData || !instruction) return res.status(400).json({ error: 'Missing data or instruction.' });

  try {
    const prompt = `Here is the current structured data for a pre-claim assessment:
${JSON.stringify(currentData, null, 2)}

The user has requested the following amendment:
"${instruction}"

Please apply these amendments to the data. Keep the exact same JSON format and keys. ONLY return the updated JSON.`;

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      temperature: 0.1,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: prompt }
      ]
    });
    
    res.json(parseAIResponse(completion.choices[0].message.content));
  } catch (err) {
    console.error('Amend error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════════
//  ANALYSE PDF — extract text → GPT-4o
// ══════════════════════════════════════════════
async function analysePDF(buffer) {
  const parsed = await pdfParse(buffer);
  const text   = parsed.text?.trim();

  if (!text || text.length < 30) {
    throw new Error('Could not extract text from the PDF. Try uploading an image instead.');
  }

  const completion = await openai.chat.completions.create({
    model: 'gpt-4o',
    temperature: 0.2,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      {
        role: 'user',
        content: `Here is the extracted text from a medical document. Analyse it and return the JSON assessment:\n\n${text.slice(0, 8000)}`
      }
    ]
  });

  return parseAIResponse(completion.choices[0].message.content);
}

// ══════════════════════════════════════════════
//  ANALYSE IMAGE — GPT-4o Vision
// ══════════════════════════════════════════════
async function analyseImage(buffer, mimetype) {
  const base64 = buffer.toString('base64');
  const dataUrl = `data:${mimetype};base64,${base64}`;

  const completion = await openai.chat.completions.create({
    model: 'gpt-4o',
    temperature: 0.2,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Analyse this medical document image and return the JSON assessment:' },
          { type: 'image_url', image_url: { url: dataUrl, detail: 'high' } }
        ]
      }
    ],
    max_tokens: 2000
  });

  return parseAIResponse(completion.choices[0].message.content);
}

// ══════════════════════════════════════════════
//  GPT-4o SYSTEM PROMPT
// ══════════════════════════════════════════════
const SYSTEM_PROMPT = `You are Ulink Pre-Claim Assessment AI.
Analyse medical cases for Singapore insurance LOG requests. Use insurer-oriented logic. 
Never provide medical advice. State that final outcome is subject to insurer review.

EXTRACT: Patient name, diagnosis, procedure, hospital, laterality, opCode, TOSP table, estimated bill, length of stay.
HOSPITAL TYPE: Private (MEH, MEN, Gleneagles, Raffles, Farrer Park, Mt Alvernia, Thomson, Crawfurd) or Government Restructured (SGH, TTSH, NUH, KTPH, CGH, SKH, NTFH). Default Private.
TOSP: Map to closest MOH Table of Surgical Procedures.

Return ONLY a valid JSON object:
{
  "date":"DD Month YYYY",
  "patientName":"", "condition":"", "procedure":"", "hospital":"", "hospType":"(Singapore – Private/Government Hospital)",
  "laterality":"", "opCode":"", "tosp_table":"", "estBill":"", "stayLength":"",
  "s1":"1-2 sentence assessment",
  "o1":"Medically Necessary", "l1":"70-90%", "r1":"1 sentence reasoning",
  "o2":"Borderline", "l2":"10-20%", "r2":"1 sentence reasoning",
  "o3":"Not Medically Necessary", "l3":"0-5%", "r3":"1 sentence reasoning",
  "pec1":"Assessment for <1 yr", "pec2":"Assessment for 1-2 yrs", "pec3":"Assessment for >2 yrs",
  "tosp_code":"CODE - Procedure", "rc_private":"SGD X,XXX - X,XXX", "rc_public":"SGD X,XXX - X,XXX", "p50":"SGD X,XXX", "p75":"", "p90":"",
  "log_rec":"One short sentence", 
  "conf_clinical":"", "conf_procedure":"", "conf_cost":"", "conf_overall":"",
  "email_subject":"Pre-Claim Assessment – [Condition/Procedure]",
  "email_body":"Hi [Name],\\n\\nPlease find attached...\\n\\nAssessment is preliminary...\\n\\nRegards\\n[Sender]"
}`;

// ══════════════════════════════════════════════
//  Parse AI response (handle markdown code blocks)
// ══════════════════════════════════════════════
function parseAIResponse(content) {
  let text = content.trim();

  // Strip markdown code blocks if present
  text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();

  // Find JSON object boundaries
  const start = text.indexOf('{');
  const end   = text.lastIndexOf('}');
  if (start !== -1 && end !== -1) {
    text = text.slice(start, end + 1);
  }

  return JSON.parse(text);
}

// ── Error handler ──
app.use((err, req, res, next) => {
  console.error(err);
  res.status(400).json({ error: err.message });
});

app.listen(port, () => {
  console.log(`✅  Ulink Pre-Claim AI Backend running on port ${port}`);
});
