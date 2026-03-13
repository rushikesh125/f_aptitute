// ProcessingHelper.ts
import fs from "node:fs"
import path from "node:path"
import { ScreenshotHelper } from "./ScreenshotHelper"
import { IProcessingHelperDeps } from "./main"
import * as axios from "axios"
import { app, BrowserWindow, dialog } from "electron"
import { OpenAI } from "openai"
import { configHelper } from "./ConfigHelper"
import Anthropic from '@anthropic-ai/sdk';

// Interface for Gemini API requests
interface GeminiMessage {
  role: string;
  parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string; } }>;
}

interface GeminiResponse {
  candidates: Array<{
    content: { parts: Array<{ text: string; }>; };
    finishReason: string;
  }>;
}

interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: Array<{
    type: 'text' | 'image';
    text?: string;
    source?: { type: 'base64'; media_type: string; data: string; };
  }>;
}

export class ProcessingHelper {
  private deps: IProcessingHelperDeps
  private screenshotHelper: ScreenshotHelper
  private openaiClient: OpenAI | null = null
  private geminiApiKey: string | null = null
  private anthropicClient: Anthropic | null = null

  // AbortControllers for API requests
  private currentProcessingAbortController: AbortController | null = null
  private currentExtraProcessingAbortController: AbortController | null = null

  constructor(deps: IProcessingHelperDeps) {
    this.deps = deps
    this.screenshotHelper = deps.getScreenshotHelper()

    // Initialize AI client based on config
    this.initializeAIClient();

    // Listen for config changes to re-initialize the AI client
    configHelper.on('config-updated', () => {
      this.initializeAIClient();
    });
  }

  /** 
   * Initialize or reinitialize the AI client with current config 
   */
  private initializeAIClient(): void {
    try {
      const config = configHelper.loadConfig();

      if (config.apiProvider === "openai") {
        if (config.apiKey) {
          this.openaiClient = new OpenAI({
            apiKey: config.apiKey,
            timeout: 60000,
            maxRetries: 2
          });
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.log("OpenAI client initialized successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, OpenAI client not initialized");
        }
      } else if (config.apiProvider === "gemini") {
        this.openaiClient = null;
        this.anthropicClient = null;
        if (config.apiKey) {
          this.geminiApiKey = config.apiKey;
          console.log("Gemini API key set successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, Gemini client not initialized");
        }
      } else if (config.apiProvider === "anthropic") {
        this.openaiClient = null;
        this.geminiApiKey = null;
        if (config.apiKey) {
          this.anthropicClient = new Anthropic({
            apiKey: config.apiKey,
            timeout: 60000,
            maxRetries: 2
          });
          console.log("Anthropic client initialized successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, Anthropic client not initialized");
        }
      }
    } catch (error) {
      console.error("Failed to initialize AI client:", error);
      this.openaiClient = null;
      this.geminiApiKey = null;
      this.anthropicClient = null;
    }
  }

  private async waitForInitialization(mainWindow: BrowserWindow): Promise<void> {
    let attempts = 0
    const maxAttempts = 50
    while (attempts < maxAttempts) {
      const isInitialized = await mainWindow.webContents.executeJavaScript("window.__IS_INITIALIZED__")
      if (isInitialized) return
      await new Promise((resolve) => setTimeout(resolve, 100))
      attempts++
    }
    throw new Error("App failed to initialize after 5 seconds")
  }

  private async getCredits(): Promise<number> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return 999
    try {
      await this.waitForInitialization(mainWindow)
      return 999
    } catch (error) {
      console.error("Error getting credits:", error)
      return 999
    }
  }

  private async getLanguage(): Promise<string> {
    try {
      const config = configHelper.loadConfig();
      if (config.language) return config.language;

      const mainWindow = this.deps.getMainWindow()
      if (mainWindow) {
        try {
          await this.waitForInitialization(mainWindow)
          const language = await mainWindow.webContents.executeJavaScript("window.__LANGUAGE__")
          if (typeof language === "string" && language) return language;
        } catch (err) {
          console.warn("Could not get language from window", err);
        }
      }
      return "python";
    } catch (error) {
      console.error("Error getting language:", error)
      return "python"
    }
  }

  public async processScreenshots(): Promise<void> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return

    const config = configHelper.loadConfig();

    if (config.apiProvider === "openai" && !this.openaiClient) {
      this.initializeAIClient();
      if (!this.openaiClient) {
        console.error("OpenAI client not initialized");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.API_KEY_INVALID);
        return;
      }
    } else if (config.apiProvider === "gemini" && !this.geminiApiKey) {
      this.initializeAIClient();
      if (!this.geminiApiKey) {
        console.error("Gemini API key not initialized");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.API_KEY_INVALID);
        return;
      }
    } else if (config.apiProvider === "anthropic" && !this.anthropicClient) {
      this.initializeAIClient();
      if (!this.anthropicClient) {
        console.error("Anthropic client not initialized");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.API_KEY_INVALID);
        return;
      }
    }

    const view = this.deps.getView()
    console.log("Processing screenshots in view:", view)

    if (view === "queue") {
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_START)
      const screenshotQueue = this.screenshotHelper.getScreenshotQueue()
      console.log("Processing main queue screenshots:", screenshotQueue)

      if (!screenshotQueue || screenshotQueue.length === 0) {
        console.log("No screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      const existingScreenshots = screenshotQueue.filter(path => fs.existsSync(path));
      if (existingScreenshots.length === 0) {
        console.log("Screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      try {
        this.currentProcessingAbortController = new AbortController()
        const { signal } = this.currentProcessingAbortController

        const screenshots = await Promise.all(
          existingScreenshots.map(async (path) => {
            try {
              return {
                path,
                preview: await this.screenshotHelper.getImagePreview(path),
                data: fs.readFileSync(path).toString('base64')
              };
            } catch (err) {
              console.error(`Error reading screenshot ${path}:`, err);
              return null;
            }
          })
        )

        const validScreenshots = screenshots.filter(Boolean);
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data");
        }

        const result = await this.processScreenshotsHelper(validScreenshots, signal)

        if (!result.success) {
          console.log("Processing failed:", result.error)
          if (result.error?.includes("API Key") || result.error?.includes("OpenAI") || result.error?.includes("Gemini")) {
            mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.API_KEY_INVALID)
          } else {
            mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, result.error)
          }
          this.deps.setView("queue")
          return
        }

        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS, result.data)
        this.deps.setView("solutions")
      } catch (error: any) {
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, error)
        console.error("Processing error:", error)
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, "Processing was canceled by the user.")
        } else {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, error.message || "Server error. Please try again.")
        }
        this.deps.setView("queue")
      } finally {
        this.currentProcessingAbortController = null
      }
    } else {
      const extraScreenshotQueue = this.screenshotHelper.getExtraScreenshotQueue()
      console.log("Processing extra queue screenshots:", extraScreenshotQueue)

      if (!extraScreenshotQueue || extraScreenshotQueue.length === 0) {
        console.log("No extra screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      const existingExtraScreenshots = extraScreenshotQueue.filter(path => fs.existsSync(path));
      if (existingExtraScreenshots.length === 0) {
        console.log("Extra screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_START)
      this.currentExtraProcessingAbortController = new AbortController()
      const { signal } = this.currentExtraProcessingAbortController

      try {
        const allPaths = [...this.screenshotHelper.getScreenshotQueue(), ...existingExtraScreenshots];
        const screenshots = await Promise.all(
          allPaths.map(async (path) => {
            try {
              if (!fs.existsSync(path)) {
                console.warn(`Screenshot file does not exist: ${path}`);
                return null;
              }
              return {
                path,
                preview: await this.screenshotHelper.getImagePreview(path),
                data: fs.readFileSync(path).toString('base64')
              };
            } catch (err) {
              console.error(`Error reading screenshot ${path}:`, err);
              return null;
            }
          })
        )

        const validScreenshots = screenshots.filter(Boolean);
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data for debugging");
        }

        console.log("Combined screenshots for processing:", validScreenshots.map((s) => s.path))
        const result = await this.processExtraScreenshotsHelper(validScreenshots, signal)

        if (result.success) {
          this.deps.setHasDebugged(true)
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_SUCCESS, result.data)
        } else {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_ERROR, result.error)
        }
      } catch (error: any) {
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_ERROR, "Extra processing was canceled by the user.")
        } else {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_ERROR, error.message)
        }
      } finally {
        this.currentExtraProcessingAbortController = null
      }
    }
  }

  // ─── SYSTEM PROMPTS ───────────────────────────────────────────────────────────

  /**
   * System/instruction prompt for the EXTRACTION step.
   * AI must classify the problem type AND return structured JSON.
   */
  private getExtractionPrompt(language: string): string {
    return `You are an expert problem analyzer for both aptitude tests and coding/DSA interviews.

Analyze the screenshot(s) and extract ALL relevant information. Then classify the problem type.

IMPORTANT: If the screenshot contains MULTIPLE aptitude questions, extract ALL of them.

Return ONLY a JSON object — no extra text, no markdown fences — with these exact fields:
{
  "problem_type": "aptitude" | "coding",
  "problem_statement": "<if single question: full problem text. If MULTIPLE questions: list all questions numbered as Q1: ... Q2: ... Q3: ... including all options>",
  "constraints": "<constraints or empty string>",
  "example_input": "<example input or empty string>",
  "example_output": "<expected output or empty string>"
}

Classification rules:
- "aptitude": math puzzles, logical reasoning, verbal ability, data interpretation, series, analogies, percentage, profit/loss, speed-distance-time, clock/calendar, seating arrangement, blood relations, etc.
- "coding": DSA problems, algorithm challenges, data structure questions, implement a function/class, time/space complexity questions, any question that requires writing code.

Preferred coding language if needed: ${language}.`;
  }

  /**
   * Solution prompt for APTITUDE problems.
   * Returns a short direct answer — no code block needed.
   */
  private getAptitudeSolutionPrompt(problemStatement: string): string {
    return `You are an expert aptitude solver. Solve the following aptitude/reasoning problem.

Problem:
${problemStatement}

Respond in this EXACT format (no deviations):

Answer: <direct answer value or option letter + value>
Reason: <one concise sentence explaining the key logic or formula used>
Formula/Trick: <short formula or trick used, if applicable; otherwise omit this line>

Keep it short and direct. No lengthy explanations.`;
  }

  /**
   * Solution prompt for CODING / DSA problems.
   * Returns working code + brief explanation.
   */
  private getCodingSolutionPrompt(problemStatement: string, constraints: string, exampleInput: string, exampleOutput: string, language: string): string {
    return `You are an expert aptitude solver.

Problem(s):
${problemStatement}

If there are MULTIPLE questions (Q1, Q2, etc.), solve ALL of them.

Respond in this EXACT format:

Q1:
Answer: <direct answer>
Reason: <one concise sentence>
Formula/Trick: <short formula if applicable>

Q2:
Answer: <direct answer>
Reason: <one concise sentence>
Formula/Trick: <short formula if applicable>

(repeat for each question)

If there is only ONE question, skip the Q1 label and just respond:
Answer: <direct answer>
Reason: <one concise sentence>
Formula/Trick: <short formula if applicable>

Keep it short and direct. No lengthy explanations.`;
  }

  /**
   * Debug prompt for APTITUDE problems.
   */
  private getAptitudeDebugPrompt(problemStatement: string): string {
    return `You are an expert aptitude coach reviewing a student's attempt.

Original Problem:
${problemStatement}

Analyze the screenshot(s) showing the student's attempt and provide feedback in this EXACT format:

### Issues Found
- <bullet: what went wrong in the attempt>

### Correct Answer
Answer: <correct direct answer>
Reason: <one concise sentence>

### Key Tip
- <one-line formula or trick to remember for this type of problem>`;
  }

  /**
   * Debug prompt for CODING / DSA problems.
   */
  private getCodingDebugPrompt(problemStatement: string, language: string): string {
    return `You are an expert coding interview coach reviewing a candidate's solution attempt.

Original Problem:
${problemStatement}

Analyze the screenshot(s) showing the candidate's code, errors, or test case failures.

Respond in this EXACT format:

### Issues Identified
- <bullet: specific bug or logical error found>

### Specific Improvements and Corrections
- <bullet: exact change needed with brief reason>

### Optimizations
- <bullet: performance improvement if any; skip section if not applicable>

### Explanation of Changes Needed
<2-3 sentences explaining why the fixes are necessary>

### Key Points
- <bullet: most important takeaway>

If showing corrected code, use a proper \`\`\`${language} code block.`;
  }

  // ─── RESPONSE FORMATTER ───────────────────────────────────────────────────────

  /**
   * Formats the raw AI response text into the shape the UI expects:
   * { code, thoughts, time_complexity, space_complexity }
   *
   * For aptitude: code field holds the "Answer + Reason" text (no actual code block).
   * For coding:   code field holds the extracted code block.
   */
  private formatSolutionResponse(responseText: string, problemType: "aptitude" | "coding"): {
    code: string;
    thoughts: string[];
    time_complexity: string;
    space_complexity: string;
  } {
    if (problemType === "aptitude") {
      // Extract Answer line
      const answerMatch = responseText.match(/Answer:\s*(.+)/i);
      const reasonMatch = responseText.match(/Reason:\s*(.+)/i);
      const formulaMatch = responseText.match(/Formula\/Trick:\s*(.+)/i);

      const answerLine = answerMatch ? `Answer: ${answerMatch[1].trim()}` : responseText.split('\n')[0];
      const reasonLine = reasonMatch ? reasonMatch[1].trim() : "";
      const formulaLine = formulaMatch ? `Trick: ${formulaMatch[1].trim()}` : "";

      // "code" field shows the answer prominently
      const codeField = answerMatch 
  ? responseText.trim()  // show full response as-is for multi-answer support
  : responseText.split('\n')[0];

      const thoughts: string[] = [];
      if (reasonLine) thoughts.push(reasonLine);
      if (formulaLine) thoughts.push(formulaLine);
      if (thoughts.length === 0) thoughts.push("Direct answer provided.");

      return {
        code: codeField,
        thoughts,
        time_complexity: "N/A",
        space_complexity: "N/A"
      };
    } else {
      // coding / DSA
      const codeMatch = responseText.match(/```(?:[a-zA-Z]*)?\s*([\s\S]*?)```/);
      const code = codeMatch ? codeMatch[1].trim() : responseText;

      const approachMatch = responseText.match(/Approach:\s*(.+)/i);
      const timeMatch = responseText.match(/Time Complexity:\s*(.+)/i);
      const spaceMatch = responseText.match(/Space Complexity:\s*(.+)/i);
      const insightMatch = responseText.match(/Key Insight:\s*(.+)/i);

      const thoughts: string[] = [];
      if (approachMatch) thoughts.push(`Approach: ${approachMatch[1].trim()}`);
      if (insightMatch) thoughts.push(`Key Insight: ${insightMatch[1].trim()}`);
      if (thoughts.length === 0) {
        // fallback: grab first few non-empty, non-code lines
        const lines = responseText
          .split('\n')
          .filter(l => l.trim() && !l.includes('```') && !l.startsWith('Time') && !l.startsWith('Space'))
          .slice(0, 3)
          .map(l => l.trim());
        thoughts.push(...lines);
      }

      return {
        code,
        thoughts: thoughts.length > 0 ? thoughts : ["Solution generated."],
        time_complexity: timeMatch ? timeMatch[1].trim() : "N/A",
        space_complexity: spaceMatch ? spaceMatch[1].trim() : "N/A"
      };
    }
  }

  // ─── MAIN PROCESSING ──────────────────────────────────────────────────────────

  private async processScreenshotsHelper(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const config = configHelper.loadConfig();
      const language = await this.getLanguage();
      const mainWindow = this.deps.getMainWindow();

      const imageDataList = screenshots.map(screenshot => screenshot.data);

      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Analyzing problem from screenshots...",
          progress: 20
        });
      }

      let problemInfo: {
        problem_type: "aptitude" | "coding";
        problem_statement: string;
        constraints: string;
        example_input: string;
        example_output: string;
      };

      const extractionPrompt = this.getExtractionPrompt(language);

      // ── EXTRACTION ──────────────────────────────────────────────────────────
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) {
          this.initializeAIClient();
          if (!this.openaiClient) {
            return { success: false, error: "OpenAI API key not configured or invalid. Please check your settings." };
          }
        }

        const messages = [
          {
            role: "system" as const,
            content: extractionPrompt
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const,
                text: "Extract and classify the problem from these screenshot(s). Return only JSON."
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `data:image/png;base64,${data}` }
              }))
            ]
          }
        ];

        const extractionResponse = await this.openaiClient.chat.completions.create({
          model: config.extractionModel || "gpt-4o",
          messages,
          max_tokens: 4000,
          temperature: 0.2
        });

        const responseText = extractionResponse.choices[0].message.content;
        const jsonText = responseText.replace(/```json|```/g, '').trim();
        problemInfo = JSON.parse(jsonText);

      } else if (config.apiProvider === "gemini") {
        if (!this.geminiApiKey) {
          return { success: false, error: "Gemini API key not configured. Please check your settings." };
        }
        try {
          const geminiMessages: GeminiMessage[] = [
            {
              role: "user",
              parts: [
                { text: extractionPrompt + "\n\nExtract and classify the problem from these screenshot(s). Return only JSON." },
                ...imageDataList.map(data => ({
                  inlineData: { mimeType: "image/png", data }
                }))
              ]
            }
          ];

          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: { temperature: 0.2, maxOutputTokens: 4000 }
            },
            { signal }
          );

          const responseData = response.data as GeminiResponse;
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("Empty response from Gemini API");
          }
          const responseText = responseData.candidates[0].content.parts[0].text;
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error) {
          console.error("Error using Gemini API:", error);
          return { success: false, error: "Failed to process with Gemini API. Please check your API key or try again later." };
        }

      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return { success: false, error: "Anthropic API key not configured. Please check your settings." };
        }
        try {
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: extractionPrompt + "\n\nExtract and classify the problem from these screenshot(s). Return only JSON."
                },
                ...imageDataList.map(data => ({
                  type: "image" as const,
                  source: { type: "base64" as const, media_type: "image/png" as const, data }
                }))
              ]
            }
          ];

          const response = await this.anthropicClient.messages.create({
            model: config.extractionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages,
            temperature: 0.2
          });

          const responseText = (response.content[0] as { type: 'text', text: string }).text;
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error: any) {
          console.error("Error using Anthropic API:", error);
          if (error.status === 429) {
            return { success: false, error: "Claude API rate limit exceeded. Please wait a few minutes before trying again." };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return { success: false, error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs." };
          }
          return { success: false, error: "Failed to process with Anthropic API. Please check your API key or try again later." };
        }
      }

      // Ensure problem_type defaults to aptitude if missing
      if (!problemInfo.problem_type) {
        problemInfo.problem_type = "aptitude";
      }

      console.log(`Detected problem type: ${problemInfo.problem_type}`);

      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: `${problemInfo.problem_type === "coding" ? "DSA/Coding" : "Aptitude"} problem detected. Generating solution...`,
          progress: 40
        });
      }

      this.deps.setProblemInfo(problemInfo);

      if (mainWindow) {
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.PROBLEM_EXTRACTED, problemInfo);
      }

      const solutionsResult = await this.generateSolutionsHelper(signal);
      if (solutionsResult.success) {
        this.screenshotHelper.clearExtraScreenshotQueue();
        mainWindow.webContents.send("processing-status", {
          message: "Solution ready",
          progress: 100
        });
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS, solutionsResult.data);
        return { success: true, data: solutionsResult.data };
      } else {
        throw new Error(solutionsResult.error || "Failed to generate solution");
      }

    } catch (error: any) {
      if (axios.isCancel(error)) {
        return { success: false, error: "Processing was canceled by the user." };
      }
      if (error?.response?.status === 401) {
        return { success: false, error: "Invalid API key. Please check your settings." };
      } else if (error?.response?.status === 429) {
        return { success: false, error: "API rate limit exceeded or insufficient credits. Please try again later." };
      } else if (error?.response?.status === 500) {
        return { success: false, error: "API server error. Please try again later." };
      }
      console.error("API Error Details:", error);
      return { success: false, error: error.message || "Failed to process screenshots. Please try again." };
    }
  }

  private async generateSolutionsHelper(signal: AbortSignal) {
    try {
      const problemInfo = this.deps.getProblemInfo() as {
        problem_type: "aptitude" | "coding";
        problem_statement: string;
        constraints: string;
        example_input: string;
        example_output: string;
      };
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      if (!problemInfo) {
        throw new Error("No problem info available");
      }

      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Generating solution...",
          progress: 60
        });
      }

      const language = await this.getLanguage();
      const problemType = problemInfo.problem_type || "aptitude";

      // Pick the right prompt based on detected problem type
      const promptText = problemType === "coding"
        ? this.getCodingSolutionPrompt(
            problemInfo.problem_statement,
            problemInfo.constraints,
            problemInfo.example_input,
            problemInfo.example_output,
            language
          )
        : this.getAptitudeSolutionPrompt(problemInfo.problem_statement);

      let responseContent: string;

      // ── SOLUTION GENERATION ─────────────────────────────────────────────────
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) {
          return { success: false, error: "OpenAI API key not configured. Please check your settings." };
        }

        const solutionResponse = await this.openaiClient.chat.completions.create({
          model: config.solutionModel || "gpt-4o",
          messages: [
            {
              role: "system",
              content: problemType === "coding"
                ? "You are an expert DSA/coding interview assistant. Always provide complete working code first, then a brief explanation."
                : "You are an expert aptitude solver. Always give a direct answer first, then a one-line explanation."
            },
            { role: "user", content: promptText }
          ],
          max_tokens: 4000,
          temperature: 0.2
        });
        responseContent = solutionResponse.choices[0].message.content;

      } else if (config.apiProvider === "gemini") {
        if (!this.geminiApiKey) {
          return { success: false, error: "Gemini API key not configured. Please check your settings." };
        }
        try {
          const geminiMessages = [
            {
              role: "user",
              parts: [{ text: promptText }]
            }
          ];

          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: { temperature: 0.2, maxOutputTokens: 4000 }
            },
            { signal }
          );

          const responseData = response.data as GeminiResponse;
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("Empty response from Gemini API");
          }
          responseContent = responseData.candidates[0].content.parts[0].text;
        } catch (error) {
          console.error("Error using Gemini API for solution:", error);
          return { success: false, error: "Failed to generate solution with Gemini API. Please check your API key or try again later." };
        }

      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return { success: false, error: "Anthropic API key not configured. Please check your settings." };
        }
        try {
          const response = await this.anthropicClient.messages.create({
            model: config.solutionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: [
              {
                role: "user" as const,
                content: [{ type: "text" as const, text: promptText }]
              }
            ],
            temperature: 0.2
          });
          responseContent = (response.content[0] as { type: 'text', text: string }).text;
        } catch (error: any) {
          console.error("Error using Anthropic API for solution:", error);
          if (error.status === 429) {
            return { success: false, error: "Claude API rate limit exceeded. Please wait a few minutes before trying again." };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return { success: false, error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs." };
          }
          return { success: false, error: "Failed to generate solution with Anthropic API. Please check your API key or try again later." };
        }
      }

      const formattedResponse = this.formatSolutionResponse(responseContent, problemType);
      return { success: true, data: formattedResponse };

    } catch (error: any) {
      if (axios.isCancel(error)) {
        return { success: false, error: "Processing was canceled by the user." };
      }
      if (error?.response?.status === 401) {
        return { success: false, error: "Invalid API key. Please check your settings." };
      } else if (error?.response?.status === 429) {
        return { success: false, error: "API rate limit exceeded or insufficient credits. Please try again later." };
      }
      console.error("Solution generation error:", error);
      return { success: false, error: error.message || "Failed to generate solution" };
    }
  }

  private async processExtraScreenshotsHelper(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const problemInfo = this.deps.getProblemInfo() as {
        problem_type: "aptitude" | "coding";
        problem_statement: string;
        constraints: string;
        example_input: string;
        example_output: string;
      };
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      if (!problemInfo) {
        throw new Error("No problem info available");
      }

      const imageDataList = screenshots.map(screenshot => screenshot.data);
      const language = await this.getLanguage();
      const problemType = problemInfo.problem_type || "aptitude";

      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Analyzing your attempt...",
          progress: 30
        });
      }

      const debugPromptText = problemType === "coding"
        ? this.getCodingDebugPrompt(problemInfo.problem_statement, language)
        : this.getAptitudeDebugPrompt(problemInfo.problem_statement);

      let debugContent: string;

      // ── DEBUG GENERATION ────────────────────────────────────────────────────
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) {
          return { success: false, error: "OpenAI API key not configured. Please check your settings." };
        }

        const messages = [
          {
            role: "system" as const,
            content: debugPromptText
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const,
                text: "Analyze my attempt from the screenshot(s) and provide structured feedback."
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `data:image/png;base64,${data}` }
              }))
            ]
          }
        ];

        if (mainWindow) {
          mainWindow.webContents.send("processing-status", {
            message: "Generating debug feedback...",
            progress: 60
          });
        }

        const debugResponse = await this.openaiClient.chat.completions.create({
          model: config.debuggingModel || "gpt-4o",
          messages,
          max_tokens: 4000,
          temperature: 0.2
        });
        debugContent = debugResponse.choices[0].message.content;

      } else if (config.apiProvider === "gemini") {
        if (!this.geminiApiKey) {
          return { success: false, error: "Gemini API key not configured. Please check your settings." };
        }
        try {
          const geminiMessages: GeminiMessage[] = [
            {
              role: "user",
              parts: [
                { text: debugPromptText + "\n\nAnalyze my attempt from the screenshot(s)." },
                ...imageDataList.map(data => ({
                  inlineData: { mimeType: "image/png", data }
                }))
              ]
            }
          ];

          if (mainWindow) {
            mainWindow.webContents.send("processing-status", {
              message: "Analyzing attempt with Gemini...",
              progress: 60
            });
          }

          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: { temperature: 0.2, maxOutputTokens: 4000 }
            },
            { signal }
          );

          const responseData = response.data as GeminiResponse;
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("Empty response from Gemini API");
          }
          debugContent = responseData.candidates[0].content.parts[0].text;
        } catch (error) {
          console.error("Error using Gemini API for debugging:", error);
          return { success: false, error: "Failed to process debug request with Gemini API. Please check your API key or try again later." };
        }

      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return { success: false, error: "Anthropic API key not configured. Please check your settings." };
        }
        try {
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: debugPromptText + "\n\nAnalyze my attempt from the screenshot(s) and provide structured feedback."
                },
                ...imageDataList.map(data => ({
                  type: "image" as const,
                  source: { type: "base64" as const, media_type: "image/png" as const, data }
                }))
              ]
            }
          ];

          const response = await this.anthropicClient.messages.create({
            model: config.debuggingModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages,
            temperature: 0.2
          });
          debugContent = (response.content[0] as { type: 'text', text: string }).text;
        } catch (error: any) {
          console.error("Error using Anthropic API for debugging:", error);
          if (error.status === 429) {
            return { success: false, error: "Claude API rate limit exceeded. Please wait a few minutes before trying again." };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return { success: false, error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs." };
          }
          return { success: false, error: "Failed to process debug request with Anthropic API. Please check your API key or try again later." };
        }
      }

      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Debug analysis complete",
          progress: 100
        });
      }

      // ── FORMAT DEBUG RESPONSE (compatible with UI) ─────────────────────────
      let extractedCode = "// See analysis below";

      if (problemType === "coding") {
        const codeMatch = debugContent.match(/```(?:[a-zA-Z]+)?\s*([\s\S]*?)```/);
        if (codeMatch && codeMatch[1]) {
          extractedCode = codeMatch[1].trim();
        }
      } else {
        // For aptitude debug, show the "Correct Answer" block as the "code" field
        const correctAnswerMatch = debugContent.match(/### Correct Answer\s*([\s\S]*?)(?=###|$)/i);
        if (correctAnswerMatch) {
          extractedCode = correctAnswerMatch[1].trim();
        }
      }

      const bulletPoints = debugContent.match(/(?:^|\n)[ ]*(?:[-*•]|\d+\.)[ ]+([^\n]+)/g);
      const thoughts = bulletPoints
        ? bulletPoints.map(p => p.replace(/^[ ]*(?:[-*•]|\d+\.)[ ]+/, '').trim()).slice(0, 5)
        : ["Check the analysis above."];

      const response = {
        code: extractedCode,
        debug_analysis: debugContent,
        thoughts,
        time_complexity: problemType === "coding" ? "N/A - Debug mode" : "N/A",
        space_complexity: problemType === "coding" ? "N/A - Debug mode" : "N/A"
      };

      return { success: true, data: response };
    } catch (error: any) {
      console.error("Debug processing error:", error);
      return { success: false, error: error.message || "Failed to process debug request" };
    }
  }

  public cancelOngoingRequests(): void {
    let wasCancelled = false
    if (this.currentProcessingAbortController) {
      this.currentProcessingAbortController.abort()
      this.currentProcessingAbortController = null
      wasCancelled = true
    }
    if (this.currentExtraProcessingAbortController) {
      this.currentExtraProcessingAbortController.abort()
      this.currentExtraProcessingAbortController = null
      wasCancelled = true
    }
    this.deps.setHasDebugged(false)
    this.deps.setProblemInfo(null)
    const mainWindow = this.deps.getMainWindow()
    if (wasCancelled && mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS)
    }
  }
}