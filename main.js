import { ChatOllama } from "@langchain/ollama";
import { START, END, MessagesAnnotation, StateGraph, MemorySaver } from "@langchain/langgraph";
import { v4 as uuidv4 } from "uuid";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { trimMessages, } from "@langchain/core/messages";

const trimmer = trimMessages({
    maxTokens: 10,
    strategy: "last",
    tokenCounter: (msgs) => msgs.length,
    includeSystem: true,
    allowPartial: false,
    startOn: "human",
});


const config = { configurable: { thread_id: uuidv4() } };

const llm = new ChatOllama({
    model: "llama3.2:3b",
    temperature: 0,
});

const promptTemplate = ChatPromptTemplate.fromMessages([
    [
        "system",
        "You talk like a pirate. Answer all questions to the best of your ability.",
    ],
    ["placeholder", "{messages}"],
]);

const callModel = async (state) => {
    const trimmedMessage = await trimmer.invoke(state.messages);
    const prompt = await promptTemplate.invoke({
        messages: trimmedMessage,
    });
    const response = await llm.invoke(prompt);

    return {
        messages: [response]
    };
};

const workflow = new StateGraph(MessagesAnnotation)
    .addNode("model", callModel)
    .addEdge(START, "model")
    .addEdge("model", END);

const app = workflow.compile({ checkpointer: new MemorySaver() });

const input = [
    {
        role: "user",
        content: "Hi! I'm Adilet.",
    },
];

const output = await app.invoke({ messages: input }, config);

const input2 = [
    {
        role: "user",
        content: "What's my name?",
    },
];

const output2 = await app.invoke({ messages: input2 }, config);
console.log(output2.messages[output2.messages.length - 1]);