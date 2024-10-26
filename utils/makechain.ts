// makeChain.ts

import { ChatOpenAI } from 'langchain/chat_models/openai';
import {
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
} from 'langchain/prompts';
import { LLMChain } from 'langchain/chains';
import type { VectorStoreRetriever } from 'langchain/vectorstores/base';
import type { Document } from 'langchain/document';
import { ConsoleCallbackHandler } from 'langchain/callbacks';

// Function to combine retrieved documents into a single context string
const combineDocumentsFn = (docs: Document[], separator = '\n\n') => {
  const serializedDocs = docs.map((doc) => doc.pageContent);
  return serializedDocs.join(separator);
};

export const makeChain = (retriever: VectorStoreRetriever) => {
  const promptTemplate = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      "You are an experienced Quality Assurance expert with expertise in defining comprehensive test cases from user stories and acceptance criteria. You have a deep understanding of software testing methodologies and best practices."
    ),
    HumanMessagePromptTemplate.fromTemplate(
      `Based on the following user story and acceptance criteria, please generate a detailed list of test cases that thoroughly cover all functional and non-functional requirements.

User Story:
{user_story}

Acceptance Criteria:
{acceptance_criteria}

For each test case, include the following:
1. **Test Case ID**: To easily identify and distinguish test cases from each other.
2. **Test Summary**: To briefly describe the case to be tested. Also known as test objective.
3. **Test Precondition**: Prerequisite, necessary condition for this test case to run.
4. **Test Steps**: Steps to execute the test case.
5. **Expected Result**: Expected result, what you want the program to do.

Ensure that your test cases are clear, concise, and cover edge cases and exception handling where appropriate.
Format your response in a structured and organized manner for easy readability.

Additional Context:
{context}

Consider any previous interactions:
{chat_history}
`
    ),
  ]);

  const model = new ChatOpenAI({
    temperature: 0,
    modelName: process.env.OPENAI_MODEL_NAME ?? 'gpt-4',
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  // Create the chain with the prompt template and model
  const chain = new LLMChain({
    llm: model,
    prompt: promptTemplate,
    // verbose: true,
    // callbacks: [new ConsoleCallbackHandler()], // For logging
  });

  return chain;
};
