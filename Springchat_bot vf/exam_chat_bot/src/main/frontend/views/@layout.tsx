import {NavLink,Outlet} from "react-router-dom";
import React from "react";
import "../styles/styles1.css"

export default function Layout() {
     return(
     <div className="p-m">
         <nav className="navbar navbar-expand-lg navbar-light bg-white py-3">
             <div className="container px-5">
                 <a className="navbar-brand" href="/"><span className="fw-bolder text-primary">Mahmoud Achraf Ahmed</span></a>
                 <button className="navbar-toggler" type="button" data-bs-toggle="collapse"
                         data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent"
                         aria-expanded="true" aria-label="Toggle navigation"><span
                     className="navbar-toggler-icon"></span></button>
                 <div className="navbar-collapse collapse show" id="navbarSupportedContent" >
                     <ul className="navbar-nav ms-auto mb-2 mb-lg-0 small fw-bolder">
                         <li className="nav-item"><a className="nav-link" href="/">Home</a></li>
                         <li className="nav-item"><a className="nav-link" href="/chat">Chat Bot</a></li>


                     </ul>
                 </div>
             </div>
         </nav>

        <main>
            <Outlet></Outlet>
        </main>
    </div>);
}