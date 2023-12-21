import * as React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGithub } from "@fortawesome/free-brands-svg-icons";
import Link from "next/link";
import moment from "moment";

const Card = ({ data }) => {
	return (
		<Link href={data.link} target="_blank" rel="noreferrer">
			<main className="card post" key={data.title}>
				<section className="card-content">
					<h6 className="subtitle is-6">{data.title}</h6>
					<p>{data.description}</p>
				</section>
				<section className="card-footer">
					<span className="date">{moment(data.date).format("MMM YYYY")}</span>
					<ul id="tags" className="is-flex-direction-row">
						{data.tags &&
							data.tags.map((tag) => {
								return (
									<li key={tag} className="tag is-warning is-light">
										{tag}
									</li>
								);
							})}
					</ul>
				</section>
			</main>
		</Link>
	);
};

export default Card;
