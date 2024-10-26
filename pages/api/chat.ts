// chat.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { makeChain } from '@/utils/makechain';
import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import type { Document } from 'langchain/document';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question, history } = req.body;

  // Only accept POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }

  const sanitizedQuestion = question.trim();

  try {
    // Initialize the vector store for RAG
    const index = pinecone.Index(PINECONE_INDEX_NAME);

    const embeddings = new OpenAIEmbeddings({
      modelName: 'text-embedding-ada-002',
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: index,
      textKey: 'text',
      namespace: PINECONE_NAME_SPACE,
    });

    // Create the retriever
    const retriever = vectorStore.asRetriever();

    // Create the chain with retriever
    const chain = makeChain(retriever);

    // Build chat history
    const pastMessages = history
      .map((message: [string, string]) => {
        return [`Human: ${message[0]}`, `Assistant: ${message[1]}`].join('\n');
      })
      .join('\n');

    // Parsing user_story and acceptance_criteria from the question
    const userStoryRegex = /User\s*Story\s*:?\s*([\s\S]*?)\s*Acceptance\s*Criteria\s*:?\s*/i;
    const acceptanceCriteriaRegex = /Acceptance\s*Criteria\s*:?\s*([\s\S]*)/i;
    const userStoryMatch = sanitizedQuestion.match(userStoryRegex);
    const acceptanceCriteriaMatch = sanitizedQuestion.match(acceptanceCriteriaRegex);

    let userStory = '';
    let acceptanceCriteria = '';

    if (userStoryMatch) {
      userStory = userStoryMatch[1].trim();
    }

    if (acceptanceCriteriaMatch) {
      acceptanceCriteria = acceptanceCriteriaMatch[1].trim();
    }

    if (!userStory || !acceptanceCriteria) {
      console.error(
        'Failed to parse user story or acceptance criteria from the question.'
      );
      return res.status(400).json({ error: 'Invalid input format.' });
    }

    console.log('Parsed user story:', userStory);
    // console.log('Parsed acceptance criteria:', acceptanceCriteria);
    // Log the acceptance criteria to verify formatting
    console.log('Parsed acceptance criteria:', JSON.stringify(acceptanceCriteria));

    // Retrieve additional context using RAG
    const retrievedDocs = await retriever.getRelevantDocuments(sanitizedQuestion);
    const context = retrievedDocs.map((doc) => doc.pageContent).join('\n\n');

    console.log('Retrieved context:', context);

    // Invoke the chain with all variables
    const response = await chain.call({
      user_story: userStory,
      acceptance_criteria: JSON.stringify(acceptanceCriteria),
      context: context,
      chat_history: pastMessages,
    });

    res.status(200).json({ text: response.text, sourceDocuments: retrievedDocs });
  } catch (error: any) {
    console.error('error', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
}
