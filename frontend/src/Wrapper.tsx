import { Outlet } from "react-router-dom";
import { Container } from "react-bootstrap";

export default function Wrapper() {
    return (
        <div className="content-wrapper">
            <div className="panel">
                <Container fluid>
                    <Outlet />
                </Container>
            </div>
        </div>
    )
}