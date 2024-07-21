import { EndpointRequestInit as EndpointRequestInit_1 } from "@vaadin/hilla-frontend";
import client_1 from "./connect-client.default.js";
async function ragChat_1(qst: string, init?: EndpointRequestInit_1): Promise<string> { return client_1.call("ChatAiService", "ragChat", { qst }, init); }
export { ragChat_1 as ragChat };
