import { render } from "preact";
import { App } from "./app";
import "./styles.css";

const root = document.getElementById("app");
if (!root) {
  throw new Error("PyAgent UI root element was not found.");
}

render(<App />, root);
