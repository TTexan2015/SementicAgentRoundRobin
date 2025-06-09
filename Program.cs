using System;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Experimental.Orchestration;

class SemanticKernelDemo
{
    static async Task Main(string[] args)
    {
        // The OpenAI connector in the Microsoft.SemanticKernel library typically
        // does not require a custom endpoint URL like Azure OpenAI. Instead, it
        // uses the default OpenAI API endpoint, which is https://api.openai.com/v1/.
        // This is handled internally by the SDK,
        // so only the API key and model ID are necessary
        string deploymentName = "gpt-4o-mini";
        string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (string.IsNullOrEmpty(apiKey))
        {
            throw new InvalidOperationException("The OpenAI API key is not set in the environment variables.");
        }

        // Create the kernel for OpenAI
        var builder = Kernel.CreateBuilder();
        builder.AddOpenAIChatCompletion(
            modelId: deploymentName,
            apiKey: apiKey
        );
        var kernel = builder.Build();

        // Define the agent prompts with input placeholders
        var newsReporterPrompt = "You are a helpful AI assistant that writes news articles based on the given topic: {{$input}}. Keep the article short.";
        var newsEditorPrompt = "You are a helpful AI assistant that rewrites and refines the news article for clarity, grammar, tone, and style. Your task is to produce a polished final draft based on the text: {{$input}}. If this is the final draft, include the message 'TERMINATE' at the end.";

        // Define the task
        var task = "Is remote work the future, or are we losing workplace culture?";

        // Create functions using CreateFunctionFromPrompt
        var newsReporterFunction = kernel.CreateFunctionFromPrompt(newsReporterPrompt);
        var newsEditorFunction = kernel.CreateFunctionFromPrompt(newsEditorPrompt);

        // Prepare the arguments
        var arguments = new KernelArguments();
        arguments["input"] = task;

        // Initial article draft by News Reporter
        var articleDraft = (await kernel.InvokeAsync(newsReporterFunction, arguments)).ToString();
        Console.WriteLine($"\nInitial NewsReporter Output:\n{articleDraft}");

        // Iterative back and forth between News Reporter and News Editor
        int maxIterations = 5;
        for (int i = 0; i < maxIterations; i++)
        {
            // Pass the draft to the News Editor for rewriting
            arguments["input"] = articleDraft;
            var editorResult = (await kernel.InvokeAsync(newsEditorFunction, arguments)).ToString();
            Console.WriteLine("==================");
            Console.WriteLine($"\nNewsEditor Rewritten Article (Iteration {i + 1}):\n{editorResult}");
            Console.WriteLine("==================");

            // Check for TERMINATE event
            if (editorResult.Contains("TERMINATE"))
            {
                Console.WriteLine("TERMINATE event detected. Ending iterations.");
                break;
            }

            // Pass the edited article back to the News Reporter for further refinement
            arguments["input"] = editorResult;
            var reporterResult = (await kernel.InvokeAsync(newsReporterFunction, arguments)).ToString();
            Console.WriteLine($"\nNewsReporter Refined Article (Iteration {i + 1}):\n{reporterResult}");
            Console.WriteLine("==================");

            // Update the article draft for the next iteration
            articleDraft = reporterResult;
        }
        Console.WriteLine("========FINAL==========");
        Console.WriteLine($"\nFinalized Article:\n{articleDraft}");
        Console.WriteLine("\nTask completed. TERMINATE");
    }
}
