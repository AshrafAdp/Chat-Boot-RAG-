import { Button, TextField } from "@vaadin/react-components";
import {useState} from "react";
import {ChatAiService} from "Frontend/generated/endpoints";
import Markdown from "react-markdown";
export default function Chat(){
    const[question,setQuestion]=useState<string>("")
    const[response,setResponse] =useState< string | undefined>("")
    async function send() {
        ChatAiService.ragChat(question).then(resp=>{ setResponse(resp) })
    }


    return (
        <div>
            <div className="d-flex flex-column h-100 bg-light">
                <main className="flex-shrink-0">
                    <section className="py-5">
                        <div className="container px-5 mb-5">
                            <div className="text-center mb-5">
                                <h1 className="display-5 fw-bolder mb-0">
                                    <span className="text-gradient d-inline">Chat Bot</span>
                                </h1>
                            </div>
                            <div className="row gx-5 justify-content-center">
                                <div className="col-lg-11 col-xl-9 col-xxl-8">
                                    <div className="bg-light rounded-4 py-5 px-4 px-md-5">
                                        <div className="text-center mb-5">
                                            <h1 className="fw-bolder">
                                                Posez vos questions librement à votre assistante virtuelle
                                            </h1>
                                        </div>
                                        <div className="row gx-5 justify-content-center">
                                            <div>
                                                <div className="form-floating mb-3">
                                                    <TextField
                                                        style={{ width: "80%" }}
                                                        onChange={(e) => setQuestion(e.target.value)}
                                                    />
                                                    <Button className="btn btn-primary btn-lg" onClick={send}>
                                                        Envoyer
                                                    </Button>

                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>

                    <section className="py-5 bg-gradient-primary-to-secondary text-white">
                        <div className="container px-5 my-5">
                            <div className="text-center">
                                <div className="card overflow-hidden shadow rounded-4 border-0">
                                    <div className="card-body p-0">
                                        <div className="d-flex align-items-center">
                                            <div className="p-5" style={{ color: "black" }}>
                                                <div>
                                                    <Markdown>
                                                        {response}
                                                    </Markdown>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>
                </main>

                <footer className="bg-white py-4 mt-auto">
                    <div className="container px-5">
                        <div className="row align-items-center justify-content-between flex-column flex-sm-row">
                            <div className="col-auto">
                                <div className="small m-0">Copyright © 2024</div>
                            </div>
                        </div>
                    </div>
                </footer>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
            <script src="js/scripts.js"></script>
        </div>
    );
}
