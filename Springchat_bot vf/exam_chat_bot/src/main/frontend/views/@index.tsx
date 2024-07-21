import "../styles/styles.css"
import "bootstrap/dist/css/bootstrap.min.css"
import React from "react";
import "../styles/styles1.css"
// @ts-ignore
import Img from '../assets/profile.png'

export default function Index(){
    return (
        <body className="d-flex flex-column h-100">
        <main className="flex-shrink-0">



            <header className="py-5">
                <div className="container px-5 pb-5">
                    <div className="row gx-5 align-items-center">
                        <div className="col-xxl-5">

                            <div className="text-center text-xxl-start">
                                <div className="badge bg-gradient-primary-to-secondary text-white mb-4">
                                    <div className="text-uppercase">Conseil · Orientation · Explications</div>
                                </div>
                                <div className="fs-3 fw-light text-muted">Communiquer avec votre virtual assistant</div>
                                <h1 className="display-3 fw-bolder mb-5"><span className="text-gradient d-inline">Posez vos questions libremente a votre assistante virtuelle</span>
                                </h1>
                                <div
                                    className="d-grid gap-3 d-sm-flex justify-content-sm-center justify-content-xxl-start mb-3">
                                    <a className="btn btn-primary btn-lg px-5 py-3 me-sm-3 fs-6 fw-bolder"
                                       href="/chat">Chat Bot</a>

                                </div>
                            </div>
                        </div>
                        <div className="col-xxl-7">

                            <div className="d-flex justify-content-center mt-5 mt-xxl-0">
                                <div className="profile bg-gradient-primary-to-secondary">

                                    <img className="profile-img" src={Img} alt="..."/>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            <section className="bg-light py-5">
                <div className="container px-5">
                    <div className="row gx-5 justify-content-center">
                        <div className="col-xxl-8">
                            <div className="text-center my-5">
                                <h2 className="display-5 fw-bolder"><span className="text-gradient d-inline">A propos de nous</span>
                                </h2>
                                <p className="lead fw-light mb-4">Mahmoud DRIOUICH - Achraf NASSIF - Ahmed REDOUANI</p>
                                <p className="text-muted"><b>Mahmoud DRIOUICH :</b> Développeur Full Stack / Ingénieur en informatique<br/>
                                    Diplômé en ingénierie informatique, développeur d'applications web, desktop et mobile utilisant diverses technologies.
                                    <br/><br/>
                                    <b>Achraf NASSIF :</b> Développeur Back-End<br/>
                                    Étudiant en cinquième année d'ingénierie en génie informatique. Passionné par la résolution de problèmes complexes et la création de solutions innovantes, avec une solide compréhension des principes fondamentaux de la programmation.
                                    <br/><br/>
                                    <b>Ahmed REDOUANI :</b> UI/UX Designer / Graphiste<br/>
                                    Étudiant en ingénierie logicielle. Exploration avec une touche artistique.</p>

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

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>

        <script src="js/scripts.js"></script>


        </body>

    )
}