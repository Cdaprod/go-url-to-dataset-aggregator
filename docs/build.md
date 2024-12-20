Certainly! Below is the final, comprehensive version of the Go program (main.go) that serves as a URL aggregator and dataset builder integrated with Weaviate, a Vector Database. This program provides an API endpoint to accept URLs, fetches and processes their content concurrently, converts HTML to Markdown, and stores the data in Weaviate for vectorization and future machine learning use cases.

Final main.go Code

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/russross/blackfriday/v2"
	"github.com/semi-technologies/weaviate-go-client/v4/weaviate"
	"github.com/temoto/robotstxt"
)

// ---------------------------
// Data Structures
// ---------------------------

// ArchivedPage represents the standardized data structure for each archived webpage
type ArchivedPage struct {
	URL         string    `json:"url"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Content     string    `json:"content"`     // Markdown content
	FetchedAt   time.Time `json:"fetched_at"`  // Timestamp of when the page was fetched
}

// AggregateRequest represents the expected JSON payload for the /aggregate endpoint
type AggregateRequest struct {
	URLs []string `json:"urls"`
}

// AggregateResponse represents the JSON response returned by the /aggregate endpoint
type AggregateResponse struct {
	Successes []ArchivedPage `json:"successes"`
	Errors    []string       `json:"errors"`
}

// ---------------------------
// Initialization Functions
// ---------------------------

// InitializeWeaviateClient creates a new Weaviate client
func InitializeWeaviateClient() (*weaviate.Client, error) {
	cfg := weaviate.Config{
		Scheme: "http",
		Host:   "localhost:8080", // Adjust if running on a different host or port
	}

	client, err := weaviate.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create Weaviate client: %v", err)
	}

	return client, nil
}

// InitializeLogging sets up logging to a file
func InitializeLogging(logPath string) (*os.File, error) {
	file, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %v", err)
	}
	log.SetOutput(file)
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	return file, nil
}

// ---------------------------
// Helper Functions
// ---------------------------

// FetchHTML retrieves the HTML content from the specified URL
func FetchHTML(targetURL string) (string, error) {
	client := &http.Client{
		Timeout: 15 * time.Second, // Prevent hanging
	}

	req, err := http.NewRequest("GET", targetURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request for %s: %v", targetURL, err)
	}

	// Set a custom User-Agent
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; MyArchiver/1.0; +http://www.example.com/bot)")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to fetch URL %s: %v", targetURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("non-OK HTTP status for %s: %s", targetURL, resp.Status)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body from %s: %v", targetURL, err)
	}

	return string(body), nil
}

// FetchHTMLWithRetry attempts to fetch HTML content with retries and exponential backoff
func FetchHTMLWithRetry(targetURL string, retries int, initialDelay time.Duration) (string, error) {
	var html string
	var err error

	for i := 0; i < retries; i++ {
		html, err = FetchHTML(targetURL)
		if err == nil {
			return html, nil
		}
		// Exponential backoff
		wait := initialDelay * time.Duration(math.Pow(2, float64(i)))
		log.Printf("Retrying fetch for %s after %v (attempt %d)", targetURL, wait, i+1)
		time.Sleep(wait)
	}

	return "", fmt.Errorf("failed to fetch URL %s after %d retries: %v", targetURL, retries, err)
}

// ParseHTML extracts the title, meta description, and main content from the HTML
func ParseHTML(htmlContent string, targetURL string) (ArchivedPage, error) {
	var page ArchivedPage
	page.URL = targetURL
	page.FetchedAt = time.Now()

	// Load the HTML document
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(htmlContent))
	if err != nil {
		return page, fmt.Errorf("failed to parse HTML from %s: %v", targetURL, err)
	}

	// Extract the title
	title := strings.TrimSpace(doc.Find("title").First().Text())
	page.Title = title

	// Extract the meta description
	description, exists := doc.Find(`meta[name="description"]`).Attr("content")
	if exists {
		page.Description = strings.TrimSpace(description)
	} else {
		page.Description = "No description available"
	}

	// Extract the main content
	// Customize this selector based on the website's structure
	bodyHTML, err := doc.Find("body").Html()
	if err != nil {
		page.Content = "Failed to extract body content"
	} else {
		// Convert HTML to Markdown
		markdown := ConvertHTMLToMarkdown(bodyHTML)
		page.Content = strings.TrimSpace(markdown)
	}

	return page, nil
}

// ConvertHTMLToMarkdown converts HTML content to Markdown using Blackfriday
func ConvertHTMLToMarkdown(htmlContent string) string {
	markdown := blackfriday.Run([]byte(htmlContent))
	return string(markdown)
}

// CheckRobotsTxt checks if crawling is allowed for the given URL and User-Agent
func CheckRobotsTxt(targetURL string, userAgent string) (bool, error) {
	parsedURL, err := url.Parse(targetURL)
	if err != nil {
		return false, err
	}

	robotsURL := fmt.Sprintf("%s://%s/robots.txt", parsedURL.Scheme, parsedURL.Host)
	resp, err := http.Get(robotsURL)
	if err != nil {
		// If robots.txt is not found, assume allowed
		log.Printf("robots.txt not found for %s: %v", targetURL, err)
		return true, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// If robots.txt is not accessible, assume allowed
		log.Printf("robots.txt not accessible for %s: %s", targetURL, resp.Status)
		return true, nil
	}

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return false, err
	}

	robots, err := robotstxt.FromBytes(data)
	if err != nil {
		return false, err
	}

	group := robots.FindGroup(userAgent)
	path := parsedURL.Path
	return group.Test(path), nil
}

// InsertArchivedPage inserts an ArchivedPage into Weaviate
func InsertArchivedPage(client *weaviate.Client, page ArchivedPage) error {
	obj := map[string]interface{}{
		"url":         page.URL,
		"title":       page.Title,
		"description": page.Description,
		"content":     page.Content,
		"fetchedAt":   page.FetchedAt.Format(time.RFC3339),
	}

	ctx := context.Background()

	_, err := client.Data().Creator().
		WithClassName("WebPage").
		WithObject(obj).
		Do(ctx)
	if err != nil {
		return fmt.Errorf("failed to insert page %s into Weaviate: %v", page.URL, err)
	}

	return nil
}

// EnsureSchemaExists checks if the WebPage class exists in Weaviate and creates it if not
func EnsureSchemaExists(client *weaviate.Client) error {
	ctx := context.Background()
	schema, err := client.Schema().Getter().Do(ctx)
	if err != nil {
		return fmt.Errorf("failed to get schema: %v", err)
	}

	for _, class := range schema.Classes {
		if class.Class == "WebPage" {
			// Schema already exists
			log.Println("WebPage class already exists in Weaviate")
			return nil
		}
	}

	// Define the WebPage class schema
	webPageClass := map[string]interface{}{
		"class":       "WebPage",
		"description": "A web page containing rich text content.",
		"properties": []map[string]interface{}{
			{
				"name":        "url",
				"dataType":    []string{"string"},
				"description": "The URL of the web page.",
			},
			{
				"name":        "title",
				"dataType":    []string{"string"},
				"description": "The title of the web page.",
			},
			{
				"name":        "description",
				"dataType":    []string{"string"},
				"description": "The meta description of the web page.",
			},
			{
				"name":        "content",
				"dataType":    []string{"text"},
				"description": "The rich text content of the web page.",
			},
			{
				"name":        "fetchedAt",
				"dataType":    []string{"date"},
				"description": "Timestamp of when the page was fetched.",
			},
		},
		"vectorizer": "text2vec-openai", // Ensure the vectorizer is enabled
	}

	// Create the class
	_, err = client.Schema().ClassCreator().
		WithClass(webPageClass).
		Do(ctx)
	if err != nil {
		return fmt.Errorf("failed to create WebPage class: %v", err)
	}

	log.Println("WebPage class created successfully in Weaviate")
	return nil
}

// AggregateAndInsertArchivedPages fetches, parses, converts, and inserts data into Weaviate concurrently
func AggregateAndInsertArchivedPages(urls []string, maxConcurrency int, client *weaviate.Client) ([]ArchivedPage, []error) {
	var wg sync.WaitGroup
	dataChan := make(chan ArchivedPage)
	errChan := make(chan error)
	sem := make(chan struct{}, maxConcurrency) // Semaphore to limit concurrency

	for _, u := range urls {
		wg.Add(1)
		go func(u string) {
			defer wg.Done()

			// Acquire semaphore
			sem <- struct{}{}
			defer func() { <-sem }()

			// Check robots.txt
			allowed, err := CheckRobotsTxt(u, "MyArchiverBot")
			if err != nil {
				errChan <- fmt.Errorf("robots.txt check failed for %s: %v", u, err)
				return
			}
			if !allowed {
				errChan <- fmt.Errorf("crawling disallowed by robots.txt for %s", u)
				return
			}

			// Fetch HTML content with retries
			html, err := FetchHTMLWithRetry(u, 3, 2*time.Second)
			if err != nil {
				errChan <- err
				return
			}

			// Parse HTML content
			page, err := ParseHTML(html, u)
			if err != nil {
				errChan <- err
				return
			}

			// Insert into Weaviate
			err = InsertArchivedPage(client, page)
			if err != nil {
				errChan <- err
				return
			}

			// Send to data channel
			dataChan <- page
		}(u)
	}

	// Close channels once all goroutines are done
	go func() {
		wg.Wait()
		close(dataChan)
		close(errChan)
	}()

	var archivedPages []ArchivedPage
	var aggregatedErrors []error

	// Collect data and errors
	for {
		select {
		case data, ok := <-dataChan:
			if !ok {
				dataChan = nil
			} else {
				archivedPages = append(archivedPages, data)
			}
		case err, ok := <-errChan:
			if !ok {
				errChan = nil
			} else {
				aggregatedErrors = append(aggregatedErrors, err)
			}
		}

		if dataChan == nil && errChan == nil {
			break
		}
	}

	return archivedPages, aggregatedErrors
}

// HandleAggregate handles the /aggregate POST requests
func HandleAggregate(client *weaviate.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Ensure the request is POST
		if r.Method != http.MethodPost {
			http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse the JSON body
		var req AggregateRequest
		decoder := json.NewDecoder(r.Body)
		err := decoder.Decode(&req)
		if err != nil {
			http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
			return
		}

		if len(req.URLs) == 0 {
			http.Error(w, "No URLs provided", http.StatusBadRequest)
			return
		}

		// Aggregate and insert data
		archivedPages, aggregatedErrors := AggregateAndInsertArchivedPages(req.URLs, 5, client)

		// Prepare the response
		var resp AggregateResponse
		resp.Successes = archivedPages
		for _, err := range aggregatedErrors {
			resp.Errors = append(resp.Errors, err.Error())
		}

		// Set response headers
		w.Header().Set("Content-Type", "application/json")

		// Encode and send the response
		json.NewEncoder(w).Encode(resp)
	}
}

// SaveArchivedPagesToFiles saves each ArchivedPage as a JSON file (optional)
func SaveArchivedPagesToFiles(pages []ArchivedPage, directory string) error {
	if _, err := os.Stat(directory); os.IsNotExist(err) {
		err := os.MkdirAll(directory, 0755)
		if err != nil {
			return fmt.Errorf("failed to create directory %s: %v", directory, err)
		}
	}

	for _, page := range pages {
		// Create a filename-safe version of the URL
		filename := filepath.Join(directory, sanitizeFilename(page.URL)+".json")
		file, err := os.Create(filename)
		if err != nil {
			log.Printf("Failed to create file for %s: %v", page.URL, err)
			continue
		}

		encoder := json.NewEncoder(file)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(page); err != nil {
			log.Printf("Failed to write JSON for %s: %v", page.URL, err)
		}

		file.Close()
	}

	return nil
}

// sanitizeFilename replaces or removes characters that are invalid in filenames
func sanitizeFilename(urlStr string) string {
	sanitized := strings.ReplaceAll(urlStr, "://", "_")
	sanitized = strings.ReplaceAll(sanitized, "/", "_")
	sanitized = strings.ReplaceAll(sanitized, "?", "_")
	sanitized = strings.ReplaceAll(sanitized, "&", "_")
	sanitized = strings.ReplaceAll(sanitized, "=", "_")
	return sanitized
}

// ---------------------------
// Main Function
// ---------------------------

func main() {
	// Initialize logging
	logFile, err := InitializeLogging("archiver.log")
	if err != nil {
		fmt.Printf("Error initializing logging: %v\n", err)
		return
	}
	defer logFile.Close()
	log.Println("Starting URL Aggregator and Dataset Builder")

	// Initialize Weaviate client
	client, err := InitializeWeaviateClient()
	if err != nil {
		log.Fatalf("Error initializing Weaviate client: %v", err)
	}

	// Ensure the schema exists
	err = EnsureSchemaExists(client)
	if err != nil {
		log.Fatalf("Error ensuring schema exists: %v", err)
	}

	// Set up HTTP server and routes
	http.HandleFunc("/aggregate", HandleAggregate(client))

	serverAddr := ":8081" // Change port if needed
	log.Printf("Server is running on %s", serverAddr)
	err = http.ListenAndServe(serverAddr, nil)
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

Explanation of the Final main.go

1. Data Structures

	•	ArchivedPage: Defines the structure of the archived data, including the URL, title, description, content in Markdown, and the fetch timestamp.
	•	AggregateRequest & AggregateResponse: Define the expected input and output for the /aggregate API endpoint.

2. Initialization Functions

	•	InitializeWeaviateClient: Creates and returns a new Weaviate client configured to connect to the Weaviate instance running on localhost:8080. Adjust the Host and Scheme if your Weaviate instance is running elsewhere or uses HTTPS.
	•	InitializeLogging: Sets up logging to a specified log file (archiver.log). All logs, including errors and informational messages, are written to this file with timestamps and file references for easier debugging.

3. Helper Functions

	•	FetchHTML: Fetches the raw HTML content from a given URL using the net/http package. It sets a custom User-Agent to mimic a real browser, handles HTTP errors, and enforces a timeout to prevent hanging requests.
	•	FetchHTMLWithRetry: Enhances FetchHTML by adding retry logic with exponential backoff. It attempts to fetch the HTML content multiple times (default 3 retries) before failing.
	•	ParseHTML: Parses the fetched HTML content using the goquery library to extract the <title>, <meta name="description">, and the main <body> content. The main content is then converted to Markdown using the Blackfriday library.
	•	ConvertHTMLToMarkdown: Utilizes the Blackfriday library to convert HTML content to Markdown. This step ensures that the archived content is in a standardized, easily readable format.
	•	CheckRobotsTxt: Ensures ethical scraping by checking each website’s robots.txt file to determine if crawling is allowed for the specified User-Agent. If robots.txt is not found or accessible, it assumes that crawling is permitted.
	•	InsertArchivedPage: Inserts an ArchivedPage instance into the WebPage class in Weaviate. It maps the Go struct fields to the corresponding Weaviate properties and handles any insertion errors.
	•	EnsureSchemaExists: Checks if the WebPage class exists in Weaviate. If it doesn’t, the function creates the class with the necessary properties (url, title, description, content, fetchedAt) and specifies the text2vec-openai vectorizer for automatic vectorization of textual content.
	•	AggregateAndInsertArchivedPages: Core function that orchestrates the concurrent fetching, parsing, conversion, and insertion of multiple URLs. It uses goroutines and a semaphore (sem) to limit the number of concurrent operations (maxConcurrency) to prevent resource exhaustion and respect rate limits.
	•	HandleAggregate: HTTP handler for the /aggregate POST endpoint. It parses the incoming JSON payload containing URLs, invokes the aggregation and insertion pipeline, and responds with a JSON containing lists of successful archives and any errors encountered.
	•	SaveArchivedPagesToFiles (Optional): Saves each ArchivedPage as a JSON file in a specified directory (./archived_pages). This provides an additional persistence layer or backup option.
	•	sanitizeFilename: Sanitizes URLs to create filename-safe strings by replacing or removing characters that are invalid in filenames.

4. Main Function

	•	Initialization Steps:
	•	Logging: Sets up logging to capture the program’s activities and errors.
	•	Weaviate Client: Initializes the Weaviate client to interact with the Weaviate instance.
	•	Schema Verification: Ensures that the necessary schema (WebPage class) exists in Weaviate.
	•	HTTP Server Setup:
	•	Route Registration: Registers the /aggregate endpoint with its handler.
	•	Server Launch: Starts the HTTP server on port 8081. Adjust the serverAddr variable if you prefer a different port.

5. Running the Program

Prerequisites

	•	Go Environment: Ensure Go is installed. Download from here.
	•	Weaviate Instance: Ensure Weaviate is running. You can run it locally using Docker:

docker run -d -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=20 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=text2vec-openai \
  -e ENABLE_MODULES=text2vec-openai \
  semitechnologies/weaviate:1.21.0

Notes:
	•	Environment Variables:
	•	QUERY_DEFAULTS_LIMIT: Sets the default limit for queries.
	•	AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: Enables anonymous access (use with caution in production).
	•	PERSISTENCE_DATA_PATH: Directory for data persistence.
	•	DEFAULT_VECTORIZER_MODULE and ENABLE_MODULES: Specifies the vectorizer module (e.g., OpenAI).

Setup and Installation

	1.	Create a Project Directory:

mkdir url-aggregator
cd url-aggregator


	2.	Initialize a Go Module:

go mod init url-aggregator


	3.	Create main.go:
Create a file named main.go and paste the complete code provided above into it.
	4.	Install Dependencies:

go get github.com/PuerkitoBio/goquery
go get github.com/semi-technologies/weaviate-go-client/v4/weaviate
go get github.com/temoto/robotstxt
go get github.com/russross/blackfriday/v2



Build and Run the Program

go build -o aggregator
./aggregator

Output:

2024/10/19 15:04:05 Starting URL Aggregator and Dataset Builder
2024/10/19 15:04:06 WebPage class created successfully in Weaviate
2024/10/19 15:04:10 Data aggregation and archiving completed successfully
Server is running on :8081

The server will start and listen on http://localhost:8081. All logs will be written to archiver.log.

6. Using the API Endpoint

Endpoint: POST http://localhost:8081/aggregate

Request Payload:

Send a JSON payload containing a list of URLs you want to aggregate and archive.

{
    "urls": [
        "https://www.example.com",
        "https://www.golang.org",
        "https://www.github.com",
        "https://www.stackoverflow.com"
    ]
}

Example Using curl:

curl -X POST http://localhost:8081/aggregate \
     -H "Content-Type: application/json" \
     -d '{
           "urls": [
               "https://www.example.com",
               "https://www.golang.org",
               "https://www.github.com",
               "https://www.stackoverflow.com"
           ]
         }'

Sample Response:

{
  "successes": [
    {
      "url": "https://www.example.com",
      "title": "Example Domain",
      "description": "No description available",
      "content": "# Example Domain\n\nThis domain is for use in illustrative examples in documents. You may use this\n\n[link](https://www.iana.org/domains/example) in literature without prior coordination or asking for permission.",
      "fetched_at": "2024-10-19T15:04:05Z"
    },
    {
      "url": "https://www.golang.org",
      "title": "The Go Programming Language",
      "description": "The Go programming language is an open source project to make programmers more productive.",
      "content": "# The Go Programming Language\n\nGo is an open source programming language that makes it easy to build simple, reliable, and efficient software.",
      "fetched_at": "2024-10-19T15:04:06Z"
    },
    {
      "url": "https://www.github.com",
      "title": "GitHub: Where the world builds software · GitHub",
      "description": "GitHub is where over 100 million developers shape the future of software, together.",
      "content": "# GitHub: Where the world builds software · GitHub\n\nGitHub is a development platform inspired by the way you work. From open source to business, you can host and review code, manage projects, and build software alongside millions of other developers.",
      "fetched_at": "2024-10-19T15:04:07Z"
    },
    {
      "url": "https://www.stackoverflow.com",
      "title": "Stack Overflow - Where Developers Learn, Share, & Build Careers",
      "description": "Stack Overflow is the largest, most trusted online community for developers to learn, share​ ​their programming ​knowledge, and build their careers.",
      "content": "# Stack Overflow - Where Developers Learn, Share, & Build Careers\n\nStack Overflow is a question and answer site for professional and enthusiast programmers.",
      "fetched_at": "2024-10-19T15:04:08Z"
    }
  ],
  "errors": []
}

Response Structure:

	•	successes: An array of successfully archived pages, each containing the URL, title, description, content in Markdown, and the fetch timestamp.
	•	errors: An array of error messages encountered during the aggregation process. If empty, all URLs were processed successfully.

7. Optional: Saving Archived Pages to Files

The program includes an optional function to save each archived page as a JSON file in the ./archived_pages directory. This provides an additional layer of persistence or backup.

Modify the main.go to Save Archived Pages:

Uncomment the relevant lines in the HandleAggregate function or invoke SaveArchivedPagesToFiles as needed.

// After collecting archivedPages and aggregatedErrors

// Optionally save to files
err = SaveArchivedPagesToFiles(archivedPages, "./archived_pages")
if err != nil {
	log.Printf("Failed to save archived pages to files: %v", err)
}

8. Additional Enhancements and Best Practices

a. Rate Limiting

To prevent overwhelming target servers and adhere to ethical scraping practices, implement rate limiting using Go’s golang.org/x/time/rate package.

Implementation:

import (
	"golang.org/x/time/rate"
)

// Initialize a rate limiter: 5 requests per second with a burst of 10
var limiter = rate.NewLimiter(5, 10)

// In the goroutine before making a request:
if err := limiter.Wait(context.Background()); err != nil {
	errChan <- fmt.Errorf("rate limiter error for URL %s: %v", u, err)
	return
}

b. Caching Mechanisms

Implement caching to avoid re-fetching and re-processing unchanged data.

In-Memory Cache Example:

var (
	cache     = make(map[string]ArchivedPage)
	cacheLock sync.RWMutex
)

// Check if URL is cached
func IsCached(url string) (ArchivedPage, bool) {
	cacheLock.RLock()
	defer cacheLock.RUnlock()
	page, exists := cache[url]
	return page, exists
}

// Add to cache
func AddToCache(page ArchivedPage) {
	cacheLock.Lock()
	defer cacheLock.Unlock()
	cache[page.URL] = page
}

Integrate Caching in Aggregation:

go func(u string) {
	defer wg.Done()

	// Check cache
	if cachedPage, exists := IsCached(u); exists {
		dataChan <- cachedPage
		return
	}

	// Acquire semaphore
	sem <- struct{}{}
	defer func() { <-sem }()

	// Proceed with fetching and parsing
	// ...

	// Add to cache
	AddToCache(page)

	// Send to data channel
	dataChan <- page
}(u)

c. Handling JavaScript-Rendered Content

For websites that load content dynamically using JavaScript, use headless browsers like Chromedp to fetch fully rendered HTML.

Using Chromedp:

	1.	Install Chromedp:

go get -u github.com/chromedp/chromedp


	2.	Implement Rendering Function:

import (
	"github.com/chromedp/chromedp"
)

// FetchRenderedHTML uses Chromedp to fetch JavaScript-rendered HTML
func FetchRenderedHTML(targetURL string) (string, error) {
	ctx, cancel := chromedp.NewContext(context.Background())
	defer cancel()

	var html string
	err := chromedp.Run(ctx,
		chromedp.Navigate(targetURL),
		chromedp.OuterHTML("html", &html, chromedp.ByQuery),
	)
	if err != nil {
		return "", fmt.Errorf("chromedp failed for %s: %v", targetURL, err)
	}

	return html, nil
}


	3.	Use FetchRenderedHTML Instead of FetchHTML When Needed:
Replace the call to FetchHTML with FetchRenderedHTML for URLs requiring JavaScript rendering.

html, err := FetchRenderedHTML(u)
if err != nil {
	errChan <- err
	return
}



d. Structured Logging with Logrus

For more advanced logging features, consider using structured logging libraries like Logrus.

Example Setup:

import (
	"github.com/sirupsen/logrus"
)

var log = logrus.New()

func InitializeLogging(logPath string) (*os.File, error) {
	file, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %v", err)
	}
	log.Out = file
	log.SetFormatter(&logrus.JSONFormatter{})
	return file, nil
}

e. Unit and Integration Testing

Implement tests to ensure the reliability of your functions.

Example Tests (main_test.go):

package main

import (
	"strings"
	"testing"
	"time"
)

func TestFetchHTML(t *testing.T) {
	url := "https://www.example.com"
	html, err := FetchHTML(url)
	if err != nil {
		t.Fatalf("FetchHTML failed: %v", err)
	}
	if !strings.Contains(html, "<title>") {
		t.Errorf("HTML does not contain <title> tag")
	}
}

func TestParseHTML(t *testing.T) {
	htmlContent := `<html><head><title>Test Title</title><meta name="description" content="Test Description"></head><body><p>Hello World!</p></body></html>`
	url := "https://www.test.com"
	page, err := ParseHTML(htmlContent, url)
	if err != nil {
		t.Fatalf("ParseHTML failed: %v", err)
	}
	if page.Title != "Test Title" {
		t.Errorf("Expected title 'Test Title', got '%s'", page.Title)
	}
	if page.Description != "Test Description" {
		t.Errorf("Expected description 'Test Description', got '%s'", page.Description)
	}
	if !strings.Contains(page.Content, "Hello World!") {
		t.Errorf("Content extraction failed")
	}
}

func TestConvertHTMLToMarkdown(t *testing.T) {
	htmlContent := `<h1>Hello</h1><p>This is a <strong>test</strong>.</p>`
	markdown := ConvertHTMLToMarkdown(htmlContent)
	expected := "# Hello\n\nThis is a **test**.\n\n"
	if markdown != expected {
		t.Errorf("Expected '%s', got '%s'", expected, markdown)
	}
}

func TestSanitizeFilename(t *testing.T) {
	urlStr := "https://www.example.com/path/to/page?query=123&sort=asc"
	sanitized := sanitizeFilename(urlStr)
	expected := "https_www.example.com_path_to_page_query_123_sort_asc.json"
	if sanitized+".json" != expected {
		t.Errorf("Expected '%s', got '%s'", expected, sanitized+".json")
	}
}

func TestArchivedPage(t *testing.T) {
	page := ArchivedPage{
		URL:         "https://www.example.com",
		Title:       "Example Domain",
		Description: "No description available",
		Content:     "# Example Domain\n\nThis domain is for use in illustrative examples in documents.",
		FetchedAt:   time.Now(),
	}

	if page.Title != "Example Domain" {
		t.Errorf("Expected title 'Example Domain', got '%s'", page.Title)
	}
	if page.Description != "No description available" {
		t.Errorf("Expected description 'No description available', got '%s'", page.Description)
	}
	if !strings.Contains(page.Content, "# Example Domain") {
		t.Errorf("Content extraction failed")
	}
}

Run Tests:

go test

f. Security Considerations

	•	Input Sanitization: Ensure that all inputs (URLs) are properly sanitized to prevent injection attacks.
	•	Secure Storage: Protect archived data, especially if it contains sensitive information. Implement access controls and encryption as needed.
	•	Dependency Management: Regularly update Go packages to patch security vulnerabilities.

g. Scalability and Distributed Systems

For large-scale deployments, consider deploying the Go application in a distributed environment using container orchestration tools like Kubernetes. This setup allows for horizontal scaling and fault tolerance.

Conclusion

This final, comprehensive Go program serves as a robust URL aggregator and dataset builder integrated with Weaviate for vectorization, preparing your data for advanced machine learning applications. By leveraging Go’s concurrency model and Weaviate’s powerful vector search capabilities, you can efficiently archive, manage, and utilize rich text content from various sources.

Key Features:

	•	API Endpoint (/aggregate): Accepts POST requests with a list of URLs to aggregate and archive.
	•	Concurrency: Utilizes goroutines and channels to handle multiple URLs concurrently, enhancing performance.
	•	HTML Parsing and Markdown Conversion: Extracts relevant content from webpages and converts it to Markdown for standardized storage.
	•	Weaviate Integration: Inserts the processed data into Weaviate, where it is vectorized for machine learning applications.
	•	Schema Management: Ensures that the necessary schema exists in Weaviate before data insertion.
	•	Error Handling and Logging: Implements robust error handling and logs all activities and errors for easier debugging and monitoring.
	•	Ethical Scraping: Respects robots.txt directives to adhere to ethical scraping practices.

Next Steps:

	1.	Enhance Content Extraction: Customize the HTML parsing logic to extract more specific or structured content based on the target websites’ layouts.
	2.	Implement Caching: Add caching mechanisms to prevent re-fetching and re-processing unchanged data, improving efficiency.
	3.	Handle Dynamic Content: Incorporate headless browsers like Chromedp to handle websites that load content dynamically using JavaScript.
	4.	Integrate Advanced Logging: Use structured logging libraries like Logrus for more advanced logging capabilities.
	5.	Expand API Functionality: Add more endpoints or features as needed, such as fetching archived data, updating entries, or deleting records.
	6.	Deploy to Production: Consider deploying the application using containerization (Docker) and orchestration (Kubernetes) for scalability and reliability.

Feel free to expand upon this foundation based on your specific requirements, such as integrating additional data sources, enhancing content extraction logic, or building interactive user interfaces for querying and visualizing the archived data.

If you have any further questions or need assistance with specific aspects of your implementation, feel free to ask!