import type { Conversation } from "../types";

const DB_NAME = "pyagent-web";
const DB_VERSION = 1;
const STORE_NAME = "conversations";
const FALLBACK_KEY = "pyagent.conversations.v1";

function fallbackRead(): Conversation[] {
  try {
    return JSON.parse(localStorage.getItem(FALLBACK_KEY) ?? "[]") as Conversation[];
  } catch {
    return [];
  }
}

function fallbackWrite(conversations: Conversation[]): void {
  try {
    localStorage.setItem(FALLBACK_KEY, JSON.stringify(conversations));
  } catch {
    // Storage can be unavailable in private or locked-down browser contexts.
  }
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function useDatabase<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const database = await openDatabase();
  try {
    return await new Promise<T>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, mode);
      const request = operation(transaction.objectStore(STORE_NAME));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      transaction.onerror = () => reject(transaction.error);
    });
  } finally {
    database.close();
  }
}

export async function listConversations(): Promise<Conversation[]> {
  if (!("indexedDB" in globalThis)) {
    return fallbackRead();
  }
  try {
    const conversations = await useDatabase<Conversation[]>(
      "readonly",
      (store) => store.getAll(),
    );
    return conversations.sort((left, right) =>
      right.updatedAt.localeCompare(left.updatedAt),
    );
  } catch {
    return fallbackRead();
  }
}

export async function saveConversation(
  conversation: Conversation,
): Promise<void> {
  if (!("indexedDB" in globalThis)) {
    const conversations = fallbackRead().filter(
      (item) => item.id !== conversation.id,
    );
    fallbackWrite([conversation, ...conversations]);
    return;
  }
  try {
    await useDatabase<IDBValidKey>("readwrite", (store) =>
      store.put(conversation),
    );
  } catch {
    const conversations = fallbackRead().filter(
      (item) => item.id !== conversation.id,
    );
    fallbackWrite([conversation, ...conversations]);
  }
}

export async function deleteConversation(id: string): Promise<void> {
  if (!("indexedDB" in globalThis)) {
    fallbackWrite(fallbackRead().filter((item) => item.id !== id));
    return;
  }
  try {
    await useDatabase<undefined>("readwrite", (store) => store.delete(id));
  } catch {
    fallbackWrite(fallbackRead().filter((item) => item.id !== id));
  }
}
