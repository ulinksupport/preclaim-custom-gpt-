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

// ── CORS — allow your Render frontend ──
app.use(cors({
  origin: [
    'https://preclaim-custom-gpt.onrender.com',
    'http://localhost',
    'http://127.0.0.1',
    /\.onrender\.com$/
  ]
}));

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

// ── Health check ──
app.get('/', (req, res) => {
  res.json({ status: 'ok', service: 'Ulink Pre-Claim AI Backend' });
});

// ══════════════════════════════════════════════
//  POST /analyse  — main endpoint
// ══════════════════════════════════════════════
app.post('/analyse', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  try {
    let result;

    if (req.file.mimetype === 'application/pdf') {
      // ── PDF: extract text, then send to GPT-4o ──
      result = await analysePDF(req.file.buffer);
    } else {
      // ── Image: send directly to GPT-4o Vision ──
      result = await analyseImage(req.file.buffer, req.file.mimetype);
    }

    res.json(result);

  } catch (err) {
    console.error('Analysis error:', err.message);
    res.status(500).json({ error: err.message || 'AI analysis failed.' });
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
const SYSTEM_PROMPT = `You are a senior pre-claim assessor at Ulink Assist, a Singapore insurance Third Party Administrator (TPA).

Your job is to analyse medical documents (referral letters, medical reports, pre-authorisation requests, discharge summaries, or investigation reports) and produce a structured pre-claim assessment.

Return ONLY a valid JSON object with this exact structure — no extra text, no markdown, no explanation:

{
  "hospital": "Full hospital name",
  "hospital_type": "Private or Public",
  "condition": "Primary diagnosis / condition",
  "procedure": "Proposed or completed procedure",
  "s1_exclusion": "Detailed general exclusion check — assess policy exclusions, waiting periods, cosmetic/preventive exclusions, and eligibility. If insufficient info, state what additional documents are needed.",
  "s2_rows": [
    {
      "outcome": "Most likely outcome (e.g. Full Recovery)",
      "likelihood": "e.g. 70-80%",
      "reasoning": "Clinical reasoning based on the document"
    },
    {
      "outcome": "Second possible outcome (e.g. Partial Recovery / Complication)",
      "likelihood": "e.g. 15-25%",
      "reasoning": "Clinical reasoning"
    },
    {
      "outcome": "Least likely outcome (e.g. No Improvement / Recurrence)",
      "likelihood": "e.g. 0-5%",
      "reasoning": "Clinical reasoning"
    }
  ],
  "s3_pec": "Pre-existing condition assessment — identify any PEC relevant to the claim, assess policy duration and PEC impact (Policy < 1yr / 1-2yr / > 2yr). State what records would confirm or rule out PEC.",
  "s4_rc": "Reasonable & Customary cost estimate — provide relevant TOSP code if applicable, estimate typical cost range for procedure at this hospital type in Singapore (SGD). Note public vs private benchmarks.",
  "s5_log": "Letter of Guarantee recommendation — state whether to issue LOG, recommended amount range (SGD), any conditions or exclusions to apply, and remarks for insurer or hospital."
}

Be specific, professional, and concise. Use Singapore insurance and medical context.`;

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
