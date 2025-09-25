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
  parts: Array<{
    text?: string;
    inlineData?: {
      mimeType: string;
      data: string;
    }
  }>;
}
interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
    finishReason: string;
  }>;
}
interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: Array<{
    type: 'text' | 'image';
    text?: string;
    source?: {
      type: 'base64';
      media_type: string;
      data: string;
    };
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
            timeout: 60000, // 60 second timeout
            maxRetries: 2   // Retry up to 2 times
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
        // Gemini client initialization
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
        // Reset other clients
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

  private async waitForInitialization(
    mainWindow: BrowserWindow
  ): Promise<void> {
    let attempts = 0
    const maxAttempts = 50 // 5 seconds total
    while (attempts < maxAttempts) {
      const isInitialized = await mainWindow.webContents.executeJavaScript(
        "window.__IS_INITIALIZED__"
      )
      if (isInitialized) return
      await new Promise((resolve) => setTimeout(resolve, 100))
      attempts++
    }
    throw new Error("App failed to initialize after 5 seconds")
  }

  private async getCredits(): Promise<number> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return 999 // Unlimited credits in this version
    try {
      await this.waitForInitialization(mainWindow)
      return 999 // Always return sufficient credits to work
    } catch (error) {
      console.error("Error getting credits:", error)
      return 999 // Unlimited credits as fallback
    }
  }

  private async getLanguage(): Promise<string> {
    try {
      // Get language from config
      const config = configHelper.loadConfig();
      if (config.language) {
        return config.language;
      }
      // Fallback to window variable if config doesn't have language
      const mainWindow = this.deps.getMainWindow()
      if (mainWindow) {
        try {
          await this.waitForInitialization(mainWindow)
          const language = await mainWindow.webContents.executeJavaScript(
            "window.__LANGUAGE__"
          )
          if (
            typeof language === "string" &&
            language !== undefined &&
            language !== null
          ) {
            return language;
          }
        } catch (err) {
          console.warn("Could not get language from window", err);
        }
      }
      // Default fallback
      return "python"; // Keeping default as 'python' for consistency, though it's less relevant now
    } catch (error) {
      console.error("Error getting language:", error)
      return "python"
    }
  }

  public async processScreenshots(): Promise<void> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return
    const config = configHelper.loadConfig();
    // First verify we have a valid AI client
    if (config.apiProvider === "openai" && !this.openaiClient) {
      this.initializeAIClient();
      if (!this.openaiClient) {
        console.error("OpenAI client not initialized");
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.API_KEY_INVALID
        );
        return;
      }
    } else if (config.apiProvider === "gemini" && !this.geminiApiKey) {
      this.initializeAIClient();
      if (!this.geminiApiKey) {
        console.error("Gemini API key not initialized");
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.API_KEY_INVALID
        );
        return;
      }
    } else if (config.apiProvider === "anthropic" && !this.anthropicClient) {
      // Add check for Anthropic client
      this.initializeAIClient();
      if (!this.anthropicClient) {
        console.error("Anthropic client not initialized");
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.API_KEY_INVALID
        );
        return;
      }
    }
    const view = this.deps.getView()
    console.log("Processing screenshots in view:", view)
    if (view === "queue") {
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_START)
      const screenshotQueue = this.screenshotHelper.getScreenshotQueue()
      console.log("Processing main queue screenshots:", screenshotQueue)
      // Check if the queue is empty
      if (!screenshotQueue || screenshotQueue.length === 0) {
        console.log("No screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }
      // Check that files actually exist
      const existingScreenshots = screenshotQueue.filter(path => fs.existsSync(path));
      if (existingScreenshots.length === 0) {
        console.log("Screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }
      try {
        // Initialize AbortController
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
        // Filter out any nulls from failed screenshots
        const validScreenshots = screenshots.filter(Boolean);
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data");
        }
        const result = await this.processScreenshotsHelper(validScreenshots, signal)
        if (!result.success) {
          console.log("Processing failed:", result.error)
          if (result.error?.includes("API Key") || result.error?.includes("OpenAI") || result.error?.includes("Gemini")) {
            mainWindow.webContents.send(
              this.deps.PROCESSING_EVENTS.API_KEY_INVALID
            )
          } else {
            mainWindow.webContents.send(
              this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
              result.error
            )
          }
          // Reset view back to queue on error
          console.log("Resetting view to queue due to error")
          this.deps.setView("queue")
          return
        }
        // Only set view to solutions if processing succeeded
        console.log("Setting view to solutions after successful processing")
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS,
          result.data
        )
        this.deps.setView("solutions")
      } catch (error: any) {
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
          error
        )
        console.error("Processing error:", error)
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
            "Processing was canceled by the user."
          )
        } else {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
            error.message || "Server error. Please try again."
          )
        }
        // Reset view back to queue on error
        console.log("Resetting view to queue due to error")
        this.deps.setView("queue")
      } finally {
        this.currentProcessingAbortController = null
      }
    } else {
      // view == 'solutions'
      const extraScreenshotQueue =
        this.screenshotHelper.getExtraScreenshotQueue()
      console.log("Processing extra queue screenshots:", extraScreenshotQueue)
      // Check if the extra queue is empty
      if (!extraScreenshotQueue || extraScreenshotQueue.length === 0) {
        console.log("No extra screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }
      // Check that files actually exist
      const existingExtraScreenshots = extraScreenshotQueue.filter(path => fs.existsSync(path));
      if (existingExtraScreenshots.length === 0) {
        console.log("Extra screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_START)
      // Initialize AbortController
      this.currentExtraProcessingAbortController = new AbortController()
      const { signal } = this.currentExtraProcessingAbortController
      try {
        // Get all screenshots (both main and extra) for processing
        const allPaths = [
          ...this.screenshotHelper.getScreenshotQueue(),
          ...existingExtraScreenshots
        ];
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
        // Filter out any nulls from failed screenshots
        const validScreenshots = screenshots.filter(Boolean);
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data for debugging");
        }
        console.log(
          "Combined screenshots for processing:",
          validScreenshots.map((s) => s.path)
        )
        const result = await this.processExtraScreenshotsHelper(
          validScreenshots,
          signal
        )
        if (result.success) {
          this.deps.setHasDebugged(true)
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_SUCCESS,
            result.data
          )
        } else {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_ERROR,
            result.error
          )
        }
      } catch (error: any) {
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_ERROR,
            "Extra processing was canceled by the user."
          )
        } else {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_ERROR,
            error.message
          )
        }
      } finally {
        this.currentExtraProcessingAbortController = null
      }
    }
  }

  private async processScreenshotsHelper(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const config = configHelper.loadConfig();
      const language = await this.getLanguage();
      const mainWindow = this.deps.getMainWindow();
      // Step 1: Extract problem info using AI Vision API (OpenAI or Gemini)
      const imageDataList = screenshots.map(screenshot => screenshot.data);
      // Update the user on progress
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Analyzing aptitude problem from screenshots...",
          progress: 20
        });
      }
      let problemInfo;
      if (config.apiProvider === "openai") {
        // Verify OpenAI client
        if (!this.openaiClient) {
          this.initializeAIClient(); // Try to reinitialize
          if (!this.openaiClient) {
            return {
              success: false,
              error: "OpenAI API key not configured or invalid. Please check your settings."
            };
          }
        }
        // Use OpenAI for processing
        const messages = [
          {
            role: "system" as const,
            content: "You are an aptitude problem interpreter. Analyze the screenshot(s) of the aptitude question and extract all relevant information. Return the information in JSON format with these fields: problem_statement, constraints, example_input, example_output. Just return the structured JSON without any other text."
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const,
                text: `Extract the aptitude problem details from these screenshots. Return in JSON format. The problem is related to aptitude tests (e.g., quantitative, logical, verbal).`
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `data:image/png;base64,${data}` }
              }))
            ]
          }
        ];
        // Send to OpenAI Vision API
        const extractionResponse = await this.openaiClient.chat.completions.create({
          model: config.extractionModel || "gpt-4o",
          messages: messages,
          max_tokens: 4000,
          temperature: 0.2
        });
        // Parse the response
        try {
          const responseText = extractionResponse.choices[0].message.content;
          // Handle when OpenAI might wrap the JSON in markdown code blocks
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error) {
          console.error("Error parsing OpenAI response:", error);
          return {
            success: false,
            error: "Failed to parse problem information. Please try again or use clearer screenshots."
          };
        }
      } else if (config.apiProvider === "gemini") {
        // Use Gemini API
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }
        try {
          // Create Gemini message structure
          const geminiMessages: GeminiMessage[] = [
            {
              role: "user",
              parts: [
                {
                  text: `You are an aptitude problem interpreter. Analyze the screenshots of the aptitude question and extract all relevant information. Return the information in JSON format with these fields: problem_statement, constraints, example_input, example_output. Just return the structured JSON without any other text. The problem is related to aptitude tests (e.g., quantitative, logical, verbal).`
                },
                ...imageDataList.map(data => ({
                  inlineData: {
                    mimeType: "image/png",
                    data: data
                  }
                }))
              ]
            }
          ];
          // Make API request to Gemini
          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.extractionModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 4000
              }
            },
            { signal }
          );
          const responseData = response.data as GeminiResponse;
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("Empty response from Gemini API");
          }
          const responseText = responseData.candidates[0].content.parts[0].text;
          // Handle when Gemini might wrap the JSON in markdown code blocks
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error) {
          console.error("Error using Gemini API:", error);
          return {
            success: false,
            error: "Failed to process with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }
        try {
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: `Extract the aptitude problem details from these screenshots. Return in JSON format with these fields: problem_statement, constraints, example_input, example_output. The problem is related to aptitude tests (e.g., quantitative, logical, verbal).`
                },
                ...imageDataList.map(data => ({
                  type: "image" as const,
                  source: {
                    type: "base64" as const,
                    media_type: "image/png" as const,
                    data: data
                  }
                }))
              ]
            }
          ];
          const response = await this.anthropicClient.messages.create({
            model: config.extractionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });
          const responseText = (response.content[0] as { type: 'text', text: string }).text;
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error: any) {
          console.error("Error using Anthropic API:", error);
          // Add specific handling for Claude's limitations
          if (error.status === 429) {
            return {
              success: false,
              error: "Claude API rate limit exceeded. Please wait a few minutes before trying again."
            };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return {
              success: false,
              error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs."
            };
          }
          return {
            success: false,
            error: "Failed to process with Anthropic API. Please check your API key or try again later."
          };
        }
      }
      // Update the user on progress
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Problem analyzed successfully. Preparing to generate solution...",
          progress: 40
        });
      }
      // Store problem info in AppState
      this.deps.setProblemInfo(problemInfo);
      // Send first success event
      if (mainWindow) {
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.PROBLEM_EXTRACTED,
          problemInfo
        );
        // Generate solutions after successful extraction
        const solutionsResult = await this.generateSolutionsHelper(signal, problemInfo);
        if (solutionsResult.success) {
          // Clear any existing extra screenshots before transitioning to solutions view
          this.screenshotHelper.clearExtraScreenshotQueue();
          // Final progress update
          mainWindow.webContents.send("processing-status", {
            message: "Aptitude solution generated successfully",
            progress: 100
          });
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS,
            solutionsResult.data
          );
          return { success: true, data: solutionsResult.data };
        } else {
          throw new Error(
            solutionsResult.error || "Failed to generate solutions"
          );
        }
      }
      return { success: false, error: "Failed to process screenshots" };
    } catch (error: any) {
      // If the request was cancelled, don't retry
      if (axios.isCancel(error)) {
        return {
          success: false,
          error: "Processing was canceled by the user."
        };
      }
      // Handle OpenAI API errors specifically
      if (error?.response?.status === 401) {
        return {
          success: false,
          error: "Invalid OpenAI API key. Please check your settings."
        };
      } else if (error?.response?.status === 429) {
        return {
          success: false,
          error: "OpenAI API rate limit exceeded or insufficient credits. Please try again later."
        };
      } else if (error?.response?.status === 500) {
        return {
          success: false,
          error: "OpenAI server error. Please try again later."
        };
      }
      console.error("API Error Details:", error);
      return {
        success: false,
        error: error.message || "Failed to process screenshots. Please try again."
      };
    }
  }

  // Modified generateSolutionsHelper to handle aptitude problems and return compatible structure
  private async generateSolutionsHelper(signal: AbortSignal, problemInfo: any) { // Now receives problemInfo as argument
    try {
      // const problemInfo = this.deps.getProblemInfo(); // Now passed as argument
      const language = await this.getLanguage();
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();
      if (!problemInfo) {
        throw new Error("No problem info available");
      }

      // Update progress status
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Creating aptitude solution with detailed explanations...",
          progress: 60
        });
      }

      // Create prompt for aptitude solution generation
      const promptText = `
Analyze and solve the following aptitude problem:
PROBLEM STATEMENT:
${problemInfo.problem_statement || problemInfo.problemDescription || problemInfo.question}

Generate the correct answer and provide a clear, step-by-step explanation of the solution process.
If the question is a multiple-choice question, please format your response as:
1. Option No. : Answer
2. Explanation: [Provide a short explanation]

If it's not multiple-choice, just provide the answer and the explanation.
`;

      let responseContent;
      if (config.apiProvider === "openai") {
        // OpenAI processing
        if (!this.openaiClient) {
          return {
            success: false,
            error: "OpenAI API key not configured. Please check your settings."
          };
        }
        // Send to OpenAI API
        const solutionResponse = await this.openaiClient.chat.completions.create({
          model: config.solutionModel || "gpt-4o",
          messages: [
            { role: "system", content: "You are an expert aptitude test assistant. Provide clear, correct solutions with detailed explanations for aptitude problems." },
            { role: "user", content: promptText }
          ],
          max_tokens: 4000,
          temperature: 0.2
        });
        responseContent = solutionResponse.choices[0].message.content;
      } else if (config.apiProvider === "gemini") {
        // Gemini processing
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }
        try {
          // Create Gemini message structure
          const geminiMessages = [
            {
              role: "user",
              parts: [
                {
                  text: `You are an Intelligent AI Assistant that helps solve Aptitude problems ( asked in Aptitude rounds of JOB interviews ) with the correct answer and explanation. Solve the following problem:
${promptText}`
                }
              ]
            }
          ];
          // Make API request to Gemini
          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.solutionModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 4000
              }
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
          return {
            success: false,
            error: "Failed to generate solution with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        // Anthropic processing
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }
        try {
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: `You are an expert aptitude test assistant. Provide a clear, correct solution with a detailed explanation for this aptitude problem:
${promptText}`
                }
              ]
            }
          ];
          // Send to Anthropic API
          const response = await this.anthropicClient.messages.create({
            model: config.solutionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });
          responseContent = (response.content[0] as { type: 'text', text: string }).text;
        } catch (error: any) {
          console.error("Error using Anthropic API for solution:", error);
          // Add specific handling for Claude's limitations
          if (error.status === 429) {
            return {
              success: false,
              error: "Claude API rate limit exceeded. Please wait a few minutes before trying again."
            };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return {
              success: false,
              error: "Your problem statement contains too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs."
            };
          }
          return {
            success: false,
            error: "Failed to generate solution with Anthropic API. Please check your API key or try again later."
          };
        }
      }

      // --- Format the response to match the expected UI structure ---
      // Place the full AI response (answer + explanation) into the 'code' field
      const formattedResponse = {
        code: responseContent, // This field will now contain the aptitude answer and explanation
        thoughts: ["aldj"], // Placeholder for thoughts
        time_complexity: "aldj", // Placeholder for time complexity
        space_complexity: "aldj" // Placeholder for space complexity
      };

      return { success: true, data: formattedResponse };
    } catch (error: any) {
      if (axios.isCancel(error)) {
        return {
          success: false,
          error: "Processing was canceled by the user."
        };
      }
      if (error?.response?.status === 401) {
        return {
          success: false,
          error: "Invalid OpenAI API key. Please check your settings."
        };
      } else if (error?.response?.status === 429) {
        return {
          success: false,
          error: "OpenAI API rate limit exceeded or insufficient credits. Please try again later."
        };
      }
      console.error("Aptitude solution generation error:", error);
      return { success: false, error: error.message || "Failed to generate aptitude solution" };
    }
  }


    private async processExtraScreenshotsHelper(
    screenshots: Array<{ path: string;  string }>,
    signal: AbortSignal
  ) {
    try {
      const problemInfo = this.deps.getProblemInfo();
      const language = await this.getLanguage(); // Keeping 'python' default for consistency, though less relevant now
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();
      if (!problemInfo) {
        throw new Error("No problem info available");
      }
      // Update progress status
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Processing aptitude review screenshots...",
          progress: 30
        });
      }
      // Prepare the images for the API call
      const imageDataList = screenshots.map(screenshot => screenshot.data);
      let debugContent;
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) {
          return {
            success: false,
            error: "OpenAI API key not configured. Please check your settings."
          };
        }
        const messages = [
          {
            role: "system" as const,
            content: `You are an aptitude test assistant helping to review and provide feedback on solutions or reasoning. Analyze these screenshots which might include notes, alternative attempts, or clarifications, and provide detailed feedback.
Your response MUST follow this exact structure with these section headers (use ### for headers):
### Issues Identified
- List each issue as a bullet point with clear explanation
### Specific Improvements and Corrections
- List specific changes needed to the reasoning or approach as bullet points
### Optimizations
- List any better approaches or shortcuts if applicable
### Explanation of Changes Needed
Here provide a clear explanation of why the changes are needed
### Key Points
- Summary bullet points of the most important takeaways
If you include examples, use proper markdown code blocks if necessary.`
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const,
                text: `I'm solving this aptitude problem: "${problemInfo.problem_statement}" in ${language}. I need help reviewing my approach or clarifying my reasoning. Here are screenshots of my work, notes, or alternative attempts. Please provide detailed feedback with:
1. What issues you found in my approach or reasoning
2. Specific improvements and corrections
3. Any optimizations or better approaches
4. A clear explanation of the changes needed`
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `image/png;base64,${data}` }
              }))
            ]
          }
        ];
        if (mainWindow) {
          mainWindow.webContents.send("processing-status", {
            message: "Analyzing work and generating feedback...",
            progress: 60
          });
        }
        const debugResponse = await this.openaiClient.chat.completions.create({
          model: config.debuggingModel || "gpt-4o",
          messages: messages,
          max_tokens: 4000,
          temperature: 0.2
        });
        debugContent = debugResponse.choices[0].message.content;
      } else if (config.apiProvider === "gemini") {
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }
        try {
          const debugPrompt = `
You are an aptitude test assistant helping to review and provide feedback on solutions or reasoning. Analyze these screenshots which might include notes, alternative attempts, or clarifications, and provide detailed feedback.
I'm solving this aptitude problem: "${problemInfo.problem_statement}" in ${language}. I need help reviewing my approach or clarifying my reasoning.
YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE WITH THESE SECTION HEADERS:
### Issues Identified
- List each issue as a bullet point with clear explanation
### Specific Improvements and Corrections
- List specific changes needed to the reasoning or approach as bullet points
### Optimizations
- List any better approaches or shortcuts if applicable
### Explanation of Changes Needed
Here provide a clear explanation of why the changes are needed
### Key Points
- Summary bullet points of the most important takeaways
If you include examples, use proper markdown code blocks if necessary.
`;
          const geminiMessages = [
            {
              role: "user",
              parts: [
                { text: debugPrompt },
                ...imageDataList.map(data => ({
                  inlineData: {
                    mimeType: "image/png",
                     data
                  }
                }))
              ]
            }
          ];
          if (mainWindow) {
            mainWindow.webContents.send("processing-status", {
              message: "Analyzing work and generating feedback with Gemini...",
              progress: 60
            });
          }
          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.debuggingModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 4000
              }
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
          return {
            success: false,
            error: "Failed to process aptitude review request with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }
        try {
          const debugPrompt = `
You are an aptitude test assistant helping to review and provide feedback on solutions or reasoning. Analyze these screenshots which might include notes, alternative attempts, or clarifications, and provide detailed feedback.
I'm solving this aptitude problem: "${problemInfo.problem_statement}" in ${language}. I need help reviewing my approach or clarifying my reasoning.
YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE WITH THESE SECTION HEADERS:
### Issues Identified
- List each issue as a bullet point with clear explanation
### Specific Improvements and Corrections
- List specific changes needed to the reasoning or approach as bullet points
### Optimizations
- List any better approaches or shortcuts if applicable
### Explanation of Changes Needed
Here provide a clear explanation of why the changes are needed
### Key Points
- Summary bullet points of the most important takeaways
If you include examples, use proper markdown code blocks if necessary.
`;
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: debugPrompt
                },
                ...imageDataList.map(data => ({
                  type: "image" as const,
                  source: {
                    type: "base64" as const,
                    media_type: "image/png" as const,
                     data
                  }
                }))
              ]
            }
          ];
          if (mainWindow) {
            mainWindow.webContents.send("processing-status", {
              message: "Analyzing work and generating feedback with Claude...",
              progress: 60
            });
          }
          const response = await this.anthropicClient.messages.create({
            model: config.debuggingModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });
          debugContent = (response.content[0] as { type: 'text', text: string }).text;
        } catch (error: any) {
          console.error("Error using Anthropic API for debugging:", error);
          // Add specific handling for Claude's limitations
          if (error.status === 429) {
            return {
              success: false,
              error: "Claude API rate limit exceeded. Please wait a few minutes before trying again."
            };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return {
              success: false,
              error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs."
            };
          }
          return {
            success: false,
            error: "Failed to process aptitude review request with Anthropic API. Please check your API key or try again later."
          };
        }
      }
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Aptitude review analysis complete",
          progress: 100
        });
      }
      // --- Parsing logic adapted for aptitude review content ---
      // Extract any code block from the response (might be formulas, notes, etc.)
      let extractedCode = "// Aptitude review - see analysis below";
      const codeMatch = debugContent.match(/```(?:[a-zA-Z]+)?([\s\S]*?)```/);
      if (codeMatch && codeMatch[1]) {
        extractedCode = codeMatch[1].trim();
      }

      // Format headers if they are missing markdown style
      let formattedDebugContent = debugContent;
      // Check if the main headers are already in markdown format (###)
      if (!debugContent.includes('### ')) {
          // If not, add markdown headers
          formattedDebugContent = debugContent
            .replace(/(?:^|
)[ ]*issues identified[ ]*(?:
|$)/i, '## Issues Identified')
            .replace(/(?:^|
)[ ]*specific improvements and corrections[ ]*(?:
|$)/i, '## Specific Improvements and Corrections')
            .replace(/(?:^|
)[ ]*optimizations[ ]*(?:
|$)/i, '## Optimizations')
            .replace(/(?:^|
)[ ]*explanation of changes needed[ ]*(?:
|$)/i, '## Explanation of Changes Needed')
            .replace(/(?:^|
)[ ]*key points[ ]*(?:
|$)/i, '## Key Points');
      }

      // Extract bullet points from the formatted content for the 'thoughts' field
      const bulletPoints = formattedDebugContent.match(/(?:^|
)[ ]*(?:[-*•]|\d+\.)[ ]+([^
]+)/g);
      const thoughts = bulletPoints
        ? bulletPoints.map(point => point.replace(/^[ ]*(?:[-*•]|\d+\.)[ ]+/, '').trim()).slice(0, 5) // Limit to 5 points
        : ["Aptitude review analysis based on your screenshots"];

      // Construct the response object maintaining the expected UI structure
      const response = {
        code: extractedCode, // Could be notes, formulas, or relevant text
        debug_analysis: formattedDebugContent, // The full formatted feedback
        thoughts: thoughts, // Key bullet points from the analysis
        time_complexity: "N/A - Aptitude Review Mode", // Placeholder
        space_complexity: "N/A - Aptitude Review Mode" // Placeholder
      };

      return { success: true,  response };
    } catch (error: any) {
      console.error("Aptitude review processing error:", error);
      return { success: false, error: error.message || "Failed to process aptitude review request" };
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