import DOMPurify from "dompurify";
import { marked } from "marked";

marked.setOptions({
  gfm: true,
  breaks: true,
});

interface MarkdownProps {
  content: string;
  streaming?: boolean;
}

export function Markdown({ content, streaming = false }: MarkdownProps) {
  const parsed = marked.parse(content || "", { async: false }) as string;
  const html = DOMPurify.sanitize(parsed, {
    USE_PROFILES: { html: true },
  });
  return (
    <div
      class={`markdown${streaming ? " markdown--streaming" : ""}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
