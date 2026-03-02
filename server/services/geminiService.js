import Groq from "groq-sdk";
import Category from "../models/category.model.js";

class GeminiCategorizationService {
  constructor() {
    this.groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
    this.model = "llama-3.3-70b-versatile";
    this.cache = new Map();
    this.domainCache = new Map();
    this.requestQueue = [];
    this.isProcessing = false;
    this.lastRequestTime = 0;
    this.minRequestInterval = 4000;
    this.retryAfter = 0;

    this.stats = {
      cacheHits: 0,
      apiCalls: 0,
      fallbacks: 0,
      rateLimited: 0,
    };
  }

  shouldUseFallback() {
    const now = Date.now();
    if (this.retryAfter > now) {
      console.log(
        `Rate limited. Using fallback for ${Math.ceil((this.retryAfter - now) / 1000)}s`,
      );
      return true;
    }
    return false;
  }

  async waitForRateLimit() {
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;

    if (timeSinceLastRequest < this.minRequestInterval) {
      const waitTime = this.minRequestInterval - timeSinceLastRequest;
      await new Promise((resolve) => setTimeout(resolve, waitTime));
    }

    this.lastRequestTime = Date.now();
  }

  async retryWithBackoff(fn, maxRetries = 3) {
    let lastError;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        await this.waitForRateLimit();
        return await fn();
      } catch (error) {
        lastError = error;

        if (error.status === 429) {
          this.stats.rateLimited++;

          const retryMatch = error.message?.match(/retry in (\d+)/i);
          const retryDelay = retryMatch
            ? parseInt(retryMatch[1]) * 1000
            : Math.pow(2, attempt + 1) * 10000;

          this.retryAfter = Date.now() + retryDelay;

          console.log(
            `Rate limited. Waiting ${retryDelay / 1000}s before retry ${attempt + 1}/${maxRetries}`,
          );

          if (attempt < maxRetries - 1) {
            await new Promise((resolve) => setTimeout(resolve, retryDelay));
          }
        } else {
          throw error;
        }
      }
    }

    throw lastError;
  }

  async categorizeActivity(url, title, domain) {
    try {
      const normTitle = (title || "").toLowerCase().trim();
      const normDomain = (domain || "").toLowerCase().trim();
      const cacheKey = `${normTitle}::${normDomain}`;

      if (this.cache.has(cacheKey)) {
        return this.cache.get(cacheKey);
      }

      const categories = await Category.find({}, "name description").lean();
      const categoryList = categories
        .map((cat) => `${cat.name}: ${cat.description}`)
        .join("\n");

      const prompt = `You are an expert web activity classifier for a screen time tracking application. Your task is to accurately categorize browsing activity to help users understand their digital habits.

## Your Role
Analyze the given webpage title and domain to determine the most appropriate category, productivity level, and relevant tags.

## Classification Guidelines

### Domain-Aware Analysis
- **Multi-purpose platforms** (YouTube, Reddit, Medium, etc.) require title analysis to determine actual content type
- **Dedicated platforms** (Netflix, Coursera, GitHub) have clearer primary purposes
- **Subdomains matter**: docs.google.com → Work/Productivity, mail.google.com → Communication

### Title Signal Patterns
| Signals | Likely Category | Productivity |
|---------|-----------------|--------------|
| course, tutorial, lecture, learn, guide, explained | Education | High (8-9) |
| vlog, reaction, funny, meme, trailer, music video | Entertainment | Low (3-4) |
| news, breaking, report, update, politics | News | Medium (5-6) |
| login, dashboard, project, sprint, deploy | Work | High (8-9) |
| buy, cart, order, shipping, sale, discount | Shopping | Low (3-4) |
| chat, message, inbox, compose, reply | Communication | Medium (5-6) |

### Language Considerations
- Titles may be in any language (Hindi, Spanish, etc.) - analyze intent regardless of language
- Non-English entertainment (music, vlogs) should still be categorized as Entertainment
- Technical terms in any language suggest Development/Education

## Examples

**Example 1:**
Title: "CHETAK | चेतक | JAGIRDAR RV | MR. MAXX"
Domain: youtube.com
→ Category: Entertainment (Hindi music video, artist name pattern)
→ Productivity: 3 (passive consumption)

**Example 2:**
Title: "Complete Machine Learning Course | Part 1 | Sheryians AI School"
Domain: youtube.com
→ Category: Education (course, structured learning content)
→ Productivity: 9 (active learning)

**Example 3:**
Title: "Pull Request #1234: Fix authentication bug"
Domain: github.com
→ Category: Development (code review, technical work)
→ Productivity: 9 (active work)

**Example 4:**
Title: "Home Feed"
Domain: instagram.com
→ Category: Social Media (social feed browsing)
→ Productivity: 2 (passive scrolling)

## Available Categories
${categoryList}

## Activity to Classify
**Title:** ${title || "Unknown Title"}
**Domain:** ${domain}
**Full URL context:** ${url || "Not provided"}

## Response Format
Respond ONLY with valid JSON, no markdown formatting:
{
  "category": "exact category name from available list",
  "confidence": 0.1-1.0,
  "reasoning": "one sentence explaining your classification logic",
  "productivityScore": 1-10,
  "tags": ["relevant", "descriptive", "tags"]
}

## Scoring Guide
- **confidence**: How certain you are (0.9+ for clear matches, 0.5-0.7 for ambiguous)
- **productivityScore**: 1-3 (distracting), 4-6 (neutral), 7-10 (productive)
- **tags**: 3-5 lowercase, descriptive tags for the content`;

      const result = await this.groq.chat.completions.create({
        messages: [
          {
            role: "system",
            content: "You are an expert web activity classifier. Respond only with valid JSON."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        model: this.model,
        temperature: 0.3,
        max_tokens: 500,
        response_format: { type: "json_object" }
      });
      
      const text = result.choices[0]?.message?.content || "{}";

      let analysis;
      try {
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          analysis = JSON.parse(jsonMatch[0]);
        } else {
          throw new Error("No JSON found in response");
        }
      } catch (parseError) {
        console.warn("Failed to parse Gemini response:", text);
        return this.getFallbackCategory(domain, title);
      }

      const validCategory = await Category.findOne({ name: analysis.category });
      if (!validCategory) {
        analysis.category = "Unknown";
      }

      analysis.confidence = Math.max(
        0.1,
        Math.min(1.0, analysis.confidence || 0.5),
      );
      analysis.productivityScore = Math.max(
        1,
        Math.min(10, analysis.productivityScore || 5),
      );
      analysis.tags = Array.isArray(analysis.tags)
        ? analysis.tags
            .map((t) => t.toLowerCase().replace(/[^a-z0-9]/gi, ""))
            .filter((t, i, arr) => t && arr.indexOf(t) === i)
            .slice(0, 5)
        : [];
      analysis.processedAt = new Date();

      if (analysis.confidence > 0.7) {
        this.cache.set(cacheKey, analysis);
        if (this.cache.size > 1000) {
          const firstKey = this.cache.keys().next().value;
          this.cache.delete(firstKey);
        }
      }

      if (analysis.confidence > 0.8 && !this.domainCache.has(normDomain)) {
        this.domainCache.set(normDomain, {
          category: analysis.category,
          productivityScore: analysis.productivityScore,
        });
      }

      return analysis;
    } catch (error) {
      if (error.status === 429) {
        const retryMatch = error.message?.match(/retry in (\d+)/i);
        const retryDelay = retryMatch ? parseInt(retryMatch[1]) * 1000 : 60000;
        this.retryAfter = Date.now() + retryDelay;
        console.log(`Rate limit hit. Falling back for ${retryDelay / 1000}s`);
      } else {
        console.error("Gemini categorization error:", error);
      }

      this.stats.fallbacks++;
      return this.getFallbackCategory(domain, title);
    }
  }

  getStats() {
    return {
      ...this.stats,
      cacheSize: this.cache.size,
      domainCacheSize: this.domainCache.size,
      isRateLimited: this.retryAfter > Date.now(),
      rateLimitEndsIn: Math.max(
        0,
        Math.ceil((this.retryAfter - Date.now()) / 1000),
      ),
    };
  }

  getFallbackCategory(domain, title = "") {
    const domainLower = domain.toLowerCase();
    const titleLower = title.toLowerCase();

    const patterns = {
      Work: [
        "github",
        "gitlab",
        "jira",
        "slack",
        "teams",
        "office",
        "google.com/drive",
        "docs.google",
        "notion",
      ],
      "Social Media": [
        "facebook",
        "twitter",
        "instagram",
        "linkedin",
        "tiktok",
        "snapchat",
        "reddit",
        "discord",
      ],
      Entertainment: [
        "youtube",
        "netflix",
        "spotify",
        "twitch",
        "hulu",
        "disney",
        "amazon.com/prime",
      ],
      Education: [
        "coursera",
        "udemy",
        "edx",
        "khanacademy",
        "stackoverflow",
        "wikipedia",
        "mit.edu",
        ".edu",
      ],
      News: [
        "cnn",
        "bbc",
        "reuters",
        "nytimes",
        "guardian",
        "news",
        "techcrunch",
      ],
      Shopping: [
        "amazon",
        "ebay",
        "shopify",
        "etsy",
        "aliexpress",
        "walmart",
        "target",
      ],
      Communication: [
        "gmail",
        "outlook",
        "mail",
        "messenger",
        "whatsapp",
        "telegram",
      ],
      Development: [
        "github",
        "gitlab",
        "stackoverflow",
        "codepen",
        "replit",
        "developer",
      ],
      Finance: [
        "bank",
        "paypal",
        "stripe",
        "coinbase",
        "robinhood",
        "mint",
        "finance",
      ],
    };

    for (const [category, keywords] of Object.entries(patterns)) {
      if (keywords.some((keyword) => domainLower.includes(keyword))) {
        return {
          category,
          confidence: 0.6,
          reasoning: `Fallback categorization based on domain pattern: ${domain}`,
          productivityScore: ["Work", "Education"].includes(category) ? 8 : 5,
          tags: [domain.split(".")[0]],
          processedAt: new Date(),
        };
      }
    }

    // YouTube-specific fallback logic
    if (domainLower.includes("youtube")) {
      if (/course|tutorial|lecture|class|how to/.test(titleLower)) {
        return {
          category: "Education",
          confidence: 0.75,
          reasoning: "YouTube title suggests educational content",
          productivityScore: 8,
          tags: ["youtube", "education"],
          processedAt: new Date(),
        };
      } else if (/vlog|trailer|funny|music|prank|reaction/.test(titleLower)) {
        return {
          category: "Entertainment",
          confidence: 0.7,
          reasoning: "YouTube title suggests entertainment content",
          productivityScore: 4,
          tags: ["youtube", "entertainment"],
          processedAt: new Date(),
        };
      }
    }

    return {
      category: "Unknown",
      confidence: 0.3,
      reasoning: "Unable to categorize based on title and domain",
      productivityScore: 5,
      tags: [domain.split(".")[0]],
      processedAt: new Date(),
    };
  }

  async batchCategorize(activities) {
    const results = [];

    for (const activity of activities) {
      try {
        const analysis = await this.categorizeActivity(
          activity.url,
          activity.title,
          activity.domain,
        );
        results.push({ ...activity, analysis });

        // Delay to avoid rate-limiting
        await new Promise((resolve) => setTimeout(resolve, 100));
      } catch (error) {
        console.error(`Error categorizing activity ${activity.url}:`, error);
        results.push({
          ...activity,
          analysis: this.getFallbackCategory(activity.domain, activity.title),
        });
      }
    }

    return results;
  }
}

export default new GeminiCategorizationService();
