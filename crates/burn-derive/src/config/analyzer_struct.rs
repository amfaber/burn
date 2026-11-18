use super::ConfigAnalyzer;
use crate::shared::field::FieldTypeAnalyzer;
use proc_macro2::{Ident, TokenStream};
use quote::quote;

/// A field's default as an expression: a string literal holds the expression's tokens, and any
/// other literal is the value itself.
fn default_value(value: &syn::Lit) -> TokenStream {
    match value {
        syn::Lit::Str(value) => value.value().parse().unwrap(),
        _ => quote! { #value },
    }
}

pub struct ConfigStructAnalyzer {
    name: Ident,
    fields_required: Vec<FieldTypeAnalyzer>,
    fields_option: Vec<FieldTypeAnalyzer>,
    fields_default: Vec<(FieldTypeAnalyzer, syn::Lit)>,
    fields_into: Vec<Ident>,
}

impl ConfigStructAnalyzer {
    pub fn new(
        name: Ident,
        fields_required: Vec<FieldTypeAnalyzer>,
        fields_option: Vec<FieldTypeAnalyzer>,
        fields_default: Vec<(FieldTypeAnalyzer, syn::Lit)>,
        fields_into: Vec<Ident>,
    ) -> Self {
        Self {
            name,
            fields_required,
            fields_option,
            fields_default,
            fields_into,
        }
    }

    fn field_input(&self, field: &FieldTypeAnalyzer) -> (TokenStream, TokenStream) {
        let name = field.ident();
        let ty = &field.field.ty;
        if self.fields_into.contains(&name) {
            (
                quote! { impl ::core::convert::Into<#ty> },
                quote! { ::core::convert::Into::into(#name) },
            )
        } else {
            (quote! { #ty }, quote! { #name })
        }
    }

    fn wrap_impl_block(&self, tokens: TokenStream) -> TokenStream {
        let name = &self.name;

        quote! {
            impl #name {
                #tokens
            }
        }
    }

    fn names(&self) -> Vec<FieldTypeAnalyzer> {
        let mut names = Vec::new();

        for field in self.fields_required.iter() {
            names.push(field.clone());
        }

        for field in self.fields_option.iter() {
            names.push(field.clone());
        }

        for (field, _) in self.fields_default.iter() {
            names.push(field.clone());
        }

        names
    }

    fn name_types(&self, names: &[FieldTypeAnalyzer]) -> Vec<TokenStream> {
        let mut name_types = Vec::new();

        for field in names.iter() {
            let name = field.ident();
            let ty = &field.field.ty;

            name_types.push(quote! {
                #name: #ty
            });
        }

        name_types
    }

    fn serde_struct_ident(&self) -> Ident {
        Ident::new(&format!("{}Serde", self.name), self.name.span())
    }

    fn gen_serialize_fn(
        &self,
        struct_name: &Ident,
        struct_gen: &TokenStream,
        names: &[FieldTypeAnalyzer],
    ) -> TokenStream {
        let name = &self.name;
        let names = names.iter().map(|name| {
            let name = name.ident();
            quote! { #name: self.#name.clone() }
        });

        quote! {
            impl burn::serde::Serialize for #name {

                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: burn::serde::Serializer {
                    #[derive(burn::serde::Serialize)]
                    #[serde(crate = "burn::serde")]
                    #struct_gen

                    let serde_state = #struct_name {
                        #(#names),*
                    };
                    serde_state.serialize(serializer)
                }
            }

        }
    }

    /// Deserializes through a mirror of the config in which a field with a default may be
    /// missing, and then takes the default that `new` gives it. A config saved before such a field
    /// was added still loads.
    fn gen_deserialize_fn(&self, struct_name: &Ident) -> TokenStream {
        let name = &self.name;
        let mut serde_fields = Vec::new();
        let mut bindings = Vec::new();
        let mut inits = Vec::new();

        for field in self.fields_required.iter().chain(self.fields_option.iter()) {
            let ident = field.ident();
            let ty = &field.field.ty;
            serde_fields.push(quote! { #ident: #ty });
            // Bound first, so that defaults can name them as they do in `new`.
            bindings.push(quote! { let #ident = serde_state.#ident; });
            inits.push(quote! { #ident });
        }

        for (field, value) in self.fields_default.iter() {
            let ident = field.ident();
            let ty = &field.field.ty;
            let value = default_value(value);
            serde_fields.push(quote! {
                #[serde(default, deserialize_with = "present")]
                #ident: Option<#ty>
            });
            inits.push(quote! {
                #ident: match serde_state.#ident {
                    Some(value) => value,
                    None => #value,
                }
            });
        }

        quote! {
            impl<'de> burn::serde::Deserialize<'de> for #name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: burn::serde::Deserializer<'de> {
                    /// A field that is in the input, which `#[serde(default)]` tells apart from a
                    /// missing one even when the field's own type is an `Option`.
                    fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
                    where
                        D: burn::serde::Deserializer<'de>,
                        T: burn::serde::Deserialize<'de>,
                    {
                        T::deserialize(deserializer).map(Some)
                    }

                    #[derive(burn::serde::Deserialize)]
                    #[serde(crate = "burn::serde")]
                    struct #struct_name {
                        #(#serde_fields),*
                    }

                    let serde_state = #struct_name::deserialize(deserializer)?;
                    #(#bindings)*
                    Ok(#name {
                        #(#inits),*
                    })
                }
            }

        }
    }

    fn gen_serde_struct(&self, names: &[TokenStream]) -> TokenStream {
        let struct_name = self.serde_struct_ident();

        quote! {
            struct #struct_name {
                #(#names),*
            }

        }
    }
}

impl ConfigAnalyzer for ConfigStructAnalyzer {
    fn gen_new_fn(&self) -> TokenStream {
        let mut body = quote! {};
        let mut args = Vec::new();

        let mut fn_docs = quote! {};
        let mut has_field_docs = false;
        let mut has_required_docs = false;
        let mut has_option_docs = false;
        let mut has_default_docs = false;
        let mut docs_header = |fn_docs: &mut TokenStream,
                               required_docs: bool,
                               option_docs: bool,
                               default_docs: bool| {
            if !has_field_docs {
                has_field_docs = true;
                fn_docs.extend(quote! {
                    #[doc = "# Arguments"]
                });
            }
            if !has_required_docs && required_docs {
                fn_docs.extend(quote! {
                    #[doc = "###### Required Arguments"]
                });
                has_required_docs = true;
            }
            if !has_option_docs && option_docs {
                fn_docs.extend(quote! {
                    #[doc = "###### Optional Arguments"]
                });
                has_option_docs = true;
            }
            if !has_default_docs && default_docs {
                fn_docs.extend(quote! {
                    #[doc = "###### Default Arguments"]
                });
                has_default_docs = true;
            }
        };

        for field in self.fields_required.iter() {
            let name = field.ident();
            let (ty, value) = self.field_input(field);
            let docs = field.docs();

            body.extend(quote! {
                #name: #value,
            });
            args.push(quote! {
                #name: #ty
            });
            docs_header(&mut fn_docs, true, false, false);
            let doc_str = format!("###### `{}`\n\n", quote!(#name));
            fn_docs.extend(quote! {
                #[doc = #doc_str]
                #(#docs)*
            });
        }

        for field in self.fields_option.iter() {
            let name = field.ident();
            let docs = field.docs();

            body.extend(quote! {
                #name: None,
            });
            docs_header(&mut fn_docs, false, true, false);
            let default_doc = "- Defaults to `None`";
            let doc_str = format!("###### `{}`\n", quote!(#name));
            fn_docs.extend(quote! {
                #[doc = #doc_str]
                #(#docs)*
                #[doc = #default_doc]
            });
        }

        for (field, value) in self.fields_default.iter() {
            let name = field.ident();
            let docs = field.docs();

            let value_expr = default_value(value);
            body.extend(quote! {
                #name: #value_expr,
            });
            docs_header(&mut fn_docs, false, false, true);
            let default_doc = format!("- Defaults to `{}`", quote!(#value));
            let doc_str = format!("###### `{}`\n", quote!(#name));
            fn_docs.extend(quote! {
                #[doc = #doc_str]
                #(#docs)*
                #[doc = #default_doc]
            });
        }

        let body = quote! {
            #[doc = "Create a new instance of the config."]
            #fn_docs
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                #(#args),*
            ) -> Self {
                Self { #body }
            }
        };
        self.wrap_impl_block(body)
    }

    fn gen_builder_fns(&self) -> TokenStream {
        let mut body = quote! {};

        for (field, value) in self.fields_default.iter() {
            let name = field.ident();
            let (ty, input) = self.field_input(field);
            let docs = field.docs();
            let default_doc = format!("- Defaults to `{}`", quote!(#value));
            let doc_str = format!(
                "Sets the value for the field [`{}`](Self::{0}).\n\n",
                quote!(#name)
            );
            let fn_docs = quote! {
                #[doc = #doc_str]
                #(#docs)*
                #[doc = #default_doc]
            };
            let fn_name = Ident::new(&format!("with_{name}"), name.span());

            body.extend(quote! {
                #fn_docs
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #input;
                    self
                }
            });
        }

        for field in self.fields_option.iter() {
            let name = field.ident();
            let (ty, input) = self.field_input(field);
            let docs = field.docs();
            let default_doc = "- Defaults to `None`";
            let doc_str = format!(
                "Sets the value for the field [`{}`](Self::{0}).\n\n",
                quote!(#name)
            );
            let fn_docs = quote! {
                #[doc = #doc_str]
                #(#docs)*
                #[doc = #default_doc]
            };
            let fn_name = Ident::new(&format!("with_{name}"), name.span());

            body.extend(quote! {
                #fn_docs
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #input;
                    self
                }
            });
        }

        self.wrap_impl_block(body)
    }

    fn gen_serde_impl(&self) -> TokenStream {
        let names = self.names();

        let struct_name = self.serde_struct_ident();
        let name_types = self.name_types(&names);
        let struct_gen = self.gen_serde_struct(&name_types);

        let serialize_gen = self.gen_serialize_fn(&struct_name, &struct_gen, &names);
        let deserialize_gen = self.gen_deserialize_fn(&struct_name);

        quote! {
            #serialize_gen
            #deserialize_gen
        }
    }

    fn gen_clone_impl(&self) -> TokenStream {
        let name = &self.name;
        let names = self.names().into_iter().map(|name| {
            let name = name.ident();
            quote! { #name: self.#name.clone() }
        });

        quote! {
            impl Clone for #name {
                fn clone(&self) -> Self {
                    Self {
                        #(#names),*
                    }
                }
            }

        }
    }

    fn gen_display_impl(&self) -> TokenStream {
        let name = &self.name;

        quote! {
            impl core::fmt::Display for #name {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.write_str(&burn::config::config_to_json(self))
                }
            }
        }
    }

    fn gen_config_impl(&self) -> TokenStream {
        let name = &self.name;

        quote! {
            impl burn::config::Config for #name {
            }
        }
    }
}
