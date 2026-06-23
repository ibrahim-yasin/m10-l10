export interface Entity {
  text: string;
  label: string;
  start_char: number;
  end_char: number;
}

export interface ExtractResponse {
  entities: Entity[];
}

export interface KGRequest {
  question: string;
}

export interface KGResponse {
  cypher: string;
  rows: Record<string, unknown>[];
  count: number;
}

export interface Citation {
  chunk_id: number;
  score: number;
}

export interface RAGRequest {
  question: string;
  k?: number;
}

export interface RAGResponse {
  answer: string;
  citations: Citation[];
  confidence: number;
}

export interface HealthResponse {
  status: string;
}